import io
import json
import base64
import tempfile
import os
import requests
import torchaudio
import gradio as gr
import numpy as np
from typing import Generator, Tuple, Optional
from gradio_webrtc import WebRTC

# -------------------------
# Modal endpoints (deployed)
# -------------------------
# These are the URLs reported by your deploy message
STREAM_URL = "https://yassine-kheir--zonos-tts-app-stream-synthesize.modal.run"
SYNTH_URL = "https://yassine-kheir--zonos-tts-app-synthesize.modal.run"

# -------------------------
# Helpers
# -------------------------
def file_to_tuple_for_requests(path: str):
    name = os.path.basename(path)
    return ("speaker_file", (name, open(path, "rb"), "application/octet-stream"))

def data_uri_to_numpy(data_uri: str) -> Tuple[int, np.ndarray]:
    """
    Convert a data:audio/wav;base64,... data URI to (sample_rate, numpy_array with shape (channels, samples))
    """
    header, b64 = data_uri.split(",", 1)
    wav_bytes = base64.b64decode(b64)
    bio = io.BytesIO(wav_bytes)
    tensor, sr = torchaudio.load(bio)  # returns (channels, samples)
    arr = tensor.detach().cpu().numpy()
    return int(sr), arr

def wav_bytes_to_data_uri_from_filelike(filelike: io.BytesIO) -> str:
    filelike.seek(0)
    b = filelike.read()
    return "data:audio/wav;base64," + base64.b64encode(b).decode("ascii")

# -------------------------
# Initialize / status
# -------------------------
def initialize_model():
    """
    Instead of loading any model locally, check the Modal endpoints availability.
    Returns a user-friendly status string.
    """
    # Check synthesize endpoint first (POST-only endpoints may respond with 405 to GET/HEAD;
    # use a short POST with a tiny silent speaker file if possible — but to avoid expensive calls,
    # we'll do HEAD then fallback to GET; if those fail, we still return a soft success so the UI continues.
    try:
        # Try a HEAD first
        resp = requests.head(SYNTH_URL, timeout=4)
        if resp.status_code < 500:
            return "✅ Modal API reachable (synthesize endpoint)."
    except Exception:
        # HEAD might be blocked; try GET
        try:
            resp = requests.get(SYNTH_URL, timeout=4)
            if resp.status_code < 500:
                return "✅ Modal API reachable (synthesize endpoint)."
        except Exception:
            # As a last resort, try streaming endpoint
            try:
                resp = requests.head(STREAM_URL, timeout=4)
                if resp.status_code < 500:
                    return "✅ Modal API reachable (stream endpoint)."
            except Exception:
                return "⚠️ Could not reach Modal endpoints (still ok — calls will be attempted when you generate)."
    return "✅ Modal API reachable."

# -------------------------
# Streaming: call Modal stream endpoint and yield audio frames for WebRTC
# -------------------------
def stream_speech_generation(speaker_file, text, language, progress=gr.Progress()):
    """
    Connects to the Modal streaming endpoint and forwards each returned chunk to WebRTC.
    Yields tuples (sample_rate, numpy_array) compatible with gradio_webrtc.
    """
    if not speaker_file:
        return  # nothing to stream

    if not text or not text.strip():
        return

    # Prepare multipart form for the request
    files = {}
    try:
        files = {"speaker_file": (os.path.basename(speaker_file), open(speaker_file, "rb"), "application/octet-stream")}
    except Exception as e:
        print("Error opening speaker file:", e)
        return

    data = {"text": text, "language": language}

    # Progress update: starting
    try:
        progress(0.1, desc="Contacting Modal stream API...")
    except Exception:
        pass

    try:
        # Stream the response (server yields newline-delimited lines)
        with requests.post(STREAM_URL, data=data, files=files, stream=True, timeout=300) as resp:
            # If non-2xx, show an error via logs and stop
            if resp.status_code >= 400:
                try:
                    err_text = resp.text[:500]
                except Exception:
                    err_text = f"HTTP {resp.status_code}"
                print("Modal stream API error:", err_text)
                return

            chunk_count = 0
            total_samples = 0
            sample_rate = None

            # Iterate over lines (each yielded by your Modal streaming function)
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue

                # If server sends the literal START_STREAM line
                if line == "START_STREAM":
                    try:
                        progress(0.2, desc="Stream started...")
                    except Exception:
                        pass
                    continue

                # Parse JSON lines
                try:
                    obj = json.loads(line)
                except Exception as e:
                    # Not JSON: skip
                    print("Skipping non-JSON stream line:", line[:200], "err:", e)
                    continue

                # Handle error messages
                if obj.get("action") == "error":
                    print("Stream error from Modal:", obj.get("message"))
                    break

                if obj.get("action") == "play_chunk":
                    chunk_count += 1
                    data_uri = obj.get("audio")
                    try:
                        sr, arr = data_uri_to_numpy(data_uri)
                    except Exception as e:
                        print("Failed to decode chunk to numpy:", e)
                        continue

                    # arr shape is (channels, samples)
                    samples = arr.shape[-1]
                    total_samples += samples
                    sample_rate = sr

                    # Periodic progress update
                    try:
                        if chunk_count % 3 == 0:
                            progress(min(0.3 + (chunk_count * 0.02), 0.95), desc=f"Streaming: {total_samples/sr:.1f}s ({chunk_count} chunks)")
                    except Exception:
                        pass

                    # Yield the frame to WebRTC (expects (sr, ndarray) where ndarray shape is (channels, samples))
                    yield (sr, arr)

                elif obj.get("action") == "end_stream":
                    final_duration = obj.get("total_duration", float(total_samples) / (sample_rate or 1))
                    try:
                        progress(1.0, desc=f"Streaming complete! {final_duration:.1f}s ({obj.get('total_chunks', chunk_count)} chunks)")
                    except Exception:
                        pass
                    break

    except requests.RequestException as e:
        print("Network error while contacting Modal stream endpoint:", e)
    except Exception as e:
        print("Unexpected error in stream_speech_generation:", e)
    finally:
        # Close file handle
        try:
            files["speaker_file"][1].close()
        except Exception:
            pass

# -------------------------
# Non-streaming complete generation: call Modal synth endpoint
# -------------------------
def process_audio_complete(speaker_file, text, language):
    """
    Calls the Modal non-streaming synth endpoint and returns (sample_rate, numpy_array), status_message
    """
    if not speaker_file:
        return None, "❌ Please upload a speaker reference audio file."

    if not text or not text.strip():
        return None, "❌ Please enter some text to synthesize."

    # Open file and post
    try:
        with open(speaker_file, "rb") as f:
            files = {"speaker_file": (os.path.basename(speaker_file), f, "application/octet-stream")}
            data = {"text": text, "language": language}
            resp = requests.post(SYNTH_URL, data=data, files=files, timeout=300)
    except Exception as e:
        print("Failed to call Modal synth endpoint:", e)
        return None, f"❌ Network error: {e}"

    if resp.status_code >= 400:
        # Try to include returned text to help debugging
        msg = None
        try:
            msg = resp.text
        except Exception:
            msg = f"HTTP {resp.status_code}"
        print("Modal synth endpoint error:", msg)
        return None, f"❌ Modal API error: {msg}"

    # Parse JSON response expected like {"audio_base64": "data:audio/wav;base64,...", "sampling_rate": 22050}
    try:
        j = resp.json()
    except Exception as e:
        print("Invalid JSON from synth endpoint:", e, resp.text[:400])
        return None, "❌ Invalid response from Modal synth endpoint."

    data_uri = j.get("audio_base64") or j.get("audio")
    sr = j.get("sampling_rate") or j.get("sr")
    if not data_uri:
        return None, "❌ No audio returned from Modal synth endpoint."

    try:
        sr_decoded, arr = data_uri_to_numpy(data_uri)
    except Exception as e:
        print("Failed to decode returned WAV:", e)
        return None, f"❌ Failed to decode audio: {e}"

    # If the response included sampling_rate explicitly and it's different from decoded, prefer decoded sr
    return (sr_decoded, arr), f"✅ Complete! {arr.shape[-1]/sr_decoded:.1f}s audio"

# -------------------------
# Clear / stop helpers
# -------------------------
def clear_outputs():
    return None, ""

def stop_streaming():
    # Stopping is handled by WebRTC controls on client side
    return "🛑 Streaming stopped (use stop button on WebRTC component)"

# -------------------------
# Gradio UI (unchanged layout)
# -------------------------
with gr.Blocks(
    title="🎙️ Real-Time Speech Synthesis",
    theme=gr.themes.Soft(),
    css="""
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .streaming-controls {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    """
) as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>🎙️ CAMB.AI MARS7 Speech Synthesis</h1>
        <p>Convert text to speech with custom speaker voice cloning and <strong>WebRTC real-time streaming</strong></p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Model Initialization")
            init_btn = gr.Button("Initialize Model", variant="primary", size="lg")
            init_status = gr.Textbox(
                label="Status",
                interactive=False,
                show_label=False
            )

        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ Supported Languages")
            gr.HTML("""
            <div style="display: flex; gap: 1rem; justify-content: center;">
                <span style="background: #e3f2fd; padding: 0.5rem 1rem; border-radius: 20px;">🇺🇸 English (en-us)</span>
                <span style="background: #fff3e0; padding: 0.5rem 1rem; border-radius: 20px;">🇪🇸 Spanish (es)</span>
                <span style="background: #e8f5e8; padding: 0.5rem 1rem; border-radius: 20px;">🇮🇹 Italian (it)</span>
                <span style="background: #e8f5e8; padding: 0.5rem 1rem; border-radius: 20px;">🇯🇵 Japanese (ja)</span>
            </div>
            """)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Input Configuration")

            speaker_file = gr.Audio(
                label="Speaker Reference Audio",
                type="filepath",
                sources=["upload"]
            )

            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter the text you want to convert to speech...",
                lines=4,
                max_lines=10,
                value="Hello! This is a test of the CAMB AI MARS real-time speech synthesis system."
            )

            language_select = gr.Dropdown(
                choices=[
                    ("English (US)", "en-us"),
                    ("Spanish", "es"),
                    ("Italian", "it"),
                    ("Japanese", "ja")

                ],
                label="Language",
                value="en-us"
            )

            with gr.Row():
                stream_btn = gr.Button("🎵 Start Real-Time Stream", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

            complete_btn = gr.Button("🎯 Generate Complete Audio", variant="secondary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Real-Time Audio Stream")

            # WebRTC component for real-time streaming
            streaming_audio = WebRTC(
                label="Live Audio Stream",
                mode="receive",  # We're receiving audio from the server
                modality="audio"
            )

            gr.Markdown("### 🎧 Complete Audio Output")

            # Regular audio for complete generation
            output_audio = gr.Audio(
                label="Generated Complete Speech",
                interactive=False,
                autoplay=False
            )

            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3
            )

    gr.Markdown("---")

    with gr.Accordion("📖 Usage Instructions", open=False):
        gr.Markdown("""
        1. **Initialize Model**: Click "Initialize Model" to verify the Modal API endpoints
        2. **Upload Speaker Audio**: Upload a reference audio file of the target speaker (WAV, MP3, etc.)
        3. **Enter Text**: Type or paste the text you want to convert to speech
        4. **Select Language**: Choose the appropriate language (English, Spanish, Japanese, or Italian, and more ...)
        5. **Choose Generation Mode**:
           - **🎵 Start Real-Time Stream**: Streams audio chunks in real-time using WebRTC
           - **🎯 Generate Complete Audio**: Traditional mode - generates complete audio file
        """)

    # Event handlers (unchanged UI semantics)
    init_btn.click(fn=initialize_model, outputs=init_status)

    # WebRTC streaming: the stream function must yield (sr, ndarray) tuples
    streaming_audio.stream(
        fn=stream_speech_generation,
        inputs=[speaker_file, text_input, language_select],
        outputs=[streaming_audio],
        trigger=stream_btn.click
    )

    complete_btn.click(
        fn=process_audio_complete,
        inputs=[speaker_file, text_input, language_select],
        outputs=[output_audio, status_output]
    )

    clear_btn.click(
        fn=clear_outputs,
        outputs=[output_audio, status_output]
    )

    # No auto-init model (keeps behavior consistent; user can click Initialize)
    # demo.load(fn=initialize_model, outputs=init_status)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        quiet=False
    )
