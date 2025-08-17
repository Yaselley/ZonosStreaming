"""
Working Gradio app: Zonos real-time TTS streaming (MP3 chunks)

Notes:
- Requires pydub (pip install pydub) and ffmpeg available on PATH.
- Assumes Zonos model API semantics from your original snippet (Zonos.from_pretrained,
  model.stream(conditioning), model.make_speaker_embedding, etc).
- Streaming can be stopped via the "Stop Stream" button.
- Streaming yields MP3 bytes encoded to data URLs (data:audio/mp3;base64,...), which Gradio's Audio component can play directly.

Author: Assistant (GPT-5 Thinking mini)
"""

import torch
import torchaudio
import gradio as gr
import io
import numpy as np
from typing import Generator, Tuple, Optional
import time
import base64
import json
import threading

from pydub import AudioSegment  # pydub uses ffmpeg to export MP3

# Import Zonos components (from your environment)
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

# ---- Helper: numpy -> MP3 bytes (robust) ----
def numpy_to_mp3(audio_array: np.ndarray, sampling_rate: int = 22050) -> bytes:
    """
    Convert a mono or stereo numpy array to MP3 bytes.
    Handles float and int arrays, normalizes floats safely.
    Returns raw mp3 bytes.
    """
    # Ensure numpy array
    audio = np.array(audio_array, copy=False)

    # If multi-channel (e.g. (2, N) or (N, 2)), pydub expects interleaved bytes for channels
    # We'll convert to int16 PCM
    if np.issubdtype(audio.dtype, np.floating):
        max_val = np.max(np.abs(audio)) if audio.size else 0.0
        if max_val == 0:
            # silence
            audio_int16 = np.zeros_like(audio, dtype=np.int16)
        else:
            audio_scaled = audio / max_val  # [-1,1]
            audio_int16 = (audio_scaled * 32767.0).astype(np.int16)
    else:
        # If already integer type, cast to int16 safely (scale/clamp if needed)
        if audio.dtype != np.int16:
            # scale down/up if bigger range
            info = np.iinfo(audio.dtype) if np.issubdtype(audio.dtype, np.integer) else None
            if info is not None:
                # normalize from dtype range to int16 range
                audio = audio.astype(np.float32) / max(abs(info.min), info.max)
                audio_int16 = (audio * 32767.0).astype(np.int16)
            else:
                audio_int16 = audio.astype(np.int16)
        else:
            audio_int16 = audio

    # pydub expects channels in frames; create AudioSegment
    # If shape is (n_samples,) -> mono. If (n_channels, n_samples) -> transpose.
    if audio_int16.ndim == 1:
        channels = 1
        raw_bytes = audio_int16.tobytes()
    elif audio_int16.ndim == 2:
        # assume shape (n_samples, channels) OR (channels, n_samples)
        if audio_int16.shape[0] <= 2 and audio_int16.shape[1] > audio_int16.shape[0]:
            # likely (channels, samples) -> transpose to (samples, channels)
            arr = audio_int16.T
        else:
            arr = audio_int16
        # Interleave channels: pydub accepts raw bytes from interleaved PCM
        channels = arr.shape[1] if arr.ndim == 2 else 1
        raw_bytes = arr.tobytes()
    else:
        # collapse extras
        arr = audio_int16.reshape(-1)
        channels = 1
        raw_bytes = arr.tobytes()

    sample_width = 2  # int16
    # Create AudioSegment from raw PCM
    audio_segment = AudioSegment(
        data=raw_bytes,
        sample_width=sample_width,
        frame_rate=int(sampling_rate),
        channels=channels
    )

    mp3_io = io.BytesIO()
    # Use a high bitrate for quality; pydub will call ffmpeg
    audio_segment.export(mp3_io, format="mp3", bitrate="192k")
    mp3_bytes = mp3_io.getvalue()
    mp3_io.close()
    return mp3_bytes

def mp3_bytes_to_data_url(mp3_bytes: bytes) -> str:
    return "data:audio/mp3;base64," + base64.b64encode(mp3_bytes).decode("ascii")


# ---- Zonos wrapper ----
class ZonosSpeechSynthesizer:
    def __init__(self, model_path="Zyphra/Zonos-v0.1-transformer", device: Optional[str] = None):
        # If a device was requested use it, else choose cuda if available
        chosen_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = chosen_device
        print(f"[Zonos] Using device: {self.device}")

        print("[Zonos] Loading model...")
        self.model = Zonos.from_pretrained(model_path, device=self.device)
        if self.device == "cuda":
            # use bfloat16 if model supports it (from your original example)
            try:
                self.model = self.model.to(self.device, dtype=torch.bfloat16)
            except Exception:
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)

        self.sampling_rate = getattr(self.model.autoencoder, "sampling_rate", 22050)
        self.model.eval()
        print("[Zonos] Model loaded.")

    def create_speaker_embedding(self, wav_path: str):
        try:
            wav, sr = torchaudio.load(wav_path)
            wav = wav.to(self.device)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            # If model API expects raw waveform tensor and sample rate:
            return self.model.make_speaker_embedding(wav, sr)
        except Exception as e:
            raise RuntimeError(f"Error creating speaker embedding: {e}")

    def synthesize_stream(self, speaker_embedding, text: str, language: str) -> Generator[np.ndarray, None, None]:
        try:
            cond_dict = make_cond_dict(text=text, speaker=speaker_embedding, language=language)
            conditioning = self.model.prepare_conditioning(cond_dict)
            # model.stream yields tensors or numpy arrays per your earlier code
            for audio_chunk in self.model.stream(conditioning):
                # ensure numpy array of shape (n_samples,) or (n_samples, channels)
                if isinstance(audio_chunk, torch.Tensor):
                    arr = audio_chunk.detach().cpu().numpy()
                else:
                    arr = np.array(audio_chunk)
                # If arr has shape (channels, samples), transpose to (samples, channels)
                if arr.ndim == 2 and arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
                    arr = arr.T
                # flatten single-channel to 1D
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                yield arr
        except Exception as e:
            raise RuntimeError(f"Synthesis error: {e}")


# ---- Global synthesizer and stop flag ----
synthesizer: Optional[ZonosSpeechSynthesizer] = None
_streaming_stop_flag = threading.Event()


def initialize_model(model_path="Zyphra/Zonos-v0.1-transformer", device: Optional[str] = None) -> str:
    global synthesizer
    try:
        synthesizer = ZonosSpeechSynthesizer(model_path=model_path, device=device)
        return "✅ Model loaded successfully!"
    except Exception as e:
        return f"❌ Error loading model: {e}"


# ---- Streaming function that returns a generator of (audio_data_url, status_text) ----
def process_audio_real_streaming(speaker_file: str, text: str, language: str):
    """
    Gradio streaming generator -> outputs must match the outputs passed to .click.
    We return a generator that yields tuples: (audio_data_url_or_none, status_text).
    """
    global synthesizer, _streaming_stop_flag

    # Clear previous stop flag
    _streaming_stop_flag.clear()

    if synthesizer is None:
        # yield nothing audio + message
        yield None, "❌ Model not initialized. Please initialize the model first."
        return

    if not speaker_file:
        yield None, "❌ Please upload a speaker reference audio file."
        return

    if not text or not text.strip():
        yield None, "❌ Please enter some text to synthesize."
        return

    try:
        yield None, "🔎 Creating speaker embedding..."
        speaker_embedding = synthesizer.create_speaker_embedding(speaker_file)

        yield None, "▶️ Starting real-time synthesis..."
        total_samples = 0
        chunk_count = 0

        for audio_chunk in synthesizer.synthesize_stream(speaker_embedding, text, language):
            if _streaming_stop_flag.is_set():
                # stop requested
                yield None, "🛑 Streaming stopped by user."
                return

            chunk_count += 1
            total_samples += int(len(audio_chunk))
            duration_so_far = total_samples / float(synthesizer.sampling_rate)

            # Convert chunk to mp3 and to data url
            mp3_bytes = numpy_to_mp3(audio_chunk, sampling_rate=synthesizer.sampling_rate)
            mp3_data_url = mp3_bytes_to_data_url(mp3_bytes)

            # Provide a concise status
            status = f"🎵 Streaming: {duration_so_far:.2f}s ({chunk_count} chunks)"

            # Yield the mp3 (data URL) and status -> these will update the Audio and Status outputs respectively
            yield mp3_data_url, status

            # a tiny sleep to let browser update; adapt as needed
            time.sleep(0.01)

        final_duration = total_samples / float(synthesizer.sampling_rate)
        yield None, f"✅ Streaming complete! {final_duration:.2f}s ({chunk_count} chunks)"
    except Exception as e:
        yield None, f"❌ Error: {str(e)}"


def process_audio_non_streaming(speaker_file: str, text: str, language: str) -> Tuple[Optional[str], Optional[Tuple[int, np.ndarray]], str]:
    """
    Non-streaming generation: returns (mp3_data_url_for_audio_component, raw_audio_tuple_for_gradio_audio_component_if_you_want, status)
    We will return the mp3 data URL for the audio component to play and a status string.
    """
    global synthesizer
    if synthesizer is None:
        return None, None, "❌ Model not initialized. Please initialize the model first."

    if not speaker_file:
        return None, None, "❌ Please upload a speaker reference audio file."

    if not text or not text.strip():
        return None, None, "❌ Please enter some text to synthesize."

    try:
        speaker_embedding = synthesizer.create_speaker_embedding(speaker_file)

        # Collect all chunks then concatenate
        chunks = []
        for chunk in synthesizer.synthesize_stream(speaker_embedding, text, language):
            chunks.append(chunk)

        if not chunks:
            return None, None, "❌ No audio generated."

        final_audio = np.concatenate(chunks).astype(np.float32)
        final_duration = len(final_audio) / float(synthesizer.sampling_rate)

        # Convert to mp3 data URL
        mp3_bytes = numpy_to_mp3(final_audio, sampling_rate=synthesizer.sampling_rate)
        mp3_data_url = mp3_bytes_to_data_url(mp3_bytes)

        # For compatibility, also return the raw tuple if desired by gradio (sample_rate, np_array).
        return mp3_data_url, (synthesizer.sampling_rate, final_audio), f"✅ Complete! {final_duration:.2f}s ({len(chunks)} chunks)"
    except Exception as e:
        return None, None, f"❌ Error: {e}"


def stop_streaming():
    """Set the stop flag so the running generator can detect it and exit."""
    _streaming_stop_flag.set()
    # The generator itself will yield the final stop status - but here return a quick response to update UI
    return None, "🛑 Stop requested. Stopping streaming..."


def clear_outputs():
    return None, "", None


# ---- Gradio UI ----
with gr.Blocks(title="🎙️ Zonos Real-Time Speech Synthesis (MP3 streaming)") as demo:
    gr.Markdown("# 🎙️ Zonos Real-Time Speech Synthesis")
    with gr.Row():
        with gr.Column(scale=1):
            init_btn = gr.Button("Initialize Model", variant="primary")
            init_status = gr.Textbox(label="Model Status", interactive=False)

            speaker_file = gr.Audio(label="Speaker Reference Audio (upload)", type="filepath")
            text_input = gr.Textbox(label="Text to synthesize", lines=4, placeholder="Type text here...")
            language_select = gr.Dropdown(choices=[("English (US)", "en-us"), ("Spanish", "es"), ("Italian", "it")], value="en-us", label="Language")

            with gr.Row():
                stream_btn = gr.Button("🎵 Real-Time Stream", variant="primary")
                stop_btn = gr.Button("🛑 Stop Stream", variant="secondary")

            with gr.Row():
                gen_btn = gr.Button("🎯 Generate Complete", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Generated Speech (plays incoming chunks)", interactive=False, autoplay=True)
            # status text
            status_output = gr.Textbox(label="Status", interactive=False)

    # Hook up events
    init_btn.click(fn=initialize_model, inputs=[], outputs=[init_status])
    # Streaming: the generator yields (audio_data_url, status_text) -> connect to (output_audio, status_output)
    stream_btn.click(fn=process_audio_real_streaming, inputs=[speaker_file, text_input, language_select], outputs=[output_audio, status_output])
    stop_btn.click(fn=stop_streaming, outputs=[output_audio, status_output])
    gen_btn.click(fn=process_audio_non_streaming, inputs=[speaker_file, text_input, language_select], outputs=[output_audio, output_audio, status_output])
    clear_btn.click(fn=clear_outputs, outputs=[output_audio, status_output, output_audio])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
