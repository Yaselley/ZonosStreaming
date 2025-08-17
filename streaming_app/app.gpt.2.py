import os
import io
import json
import base64
import tempfile
from typing import Generator, Tuple

import torch
import torchaudio
import numpy as np
import soundfile as sf
import requests
import gradio as gr

# Import Gradio WebRTC for streaming
from gradio_webrtc import WebRTC

# ------------------------------------------------------------------
# Configuration: set SERVER_URL to your Modal/FastAPI server
# ------------------------------------------------------------------
# SERVER_URL = os.environ.get("ZONOS_SERVER", "http://127.0.0.1:8000")
SERVER_URL = "https://modal.com/apps/yassine-kheir/main/deployed/zonos-api"

# ------------------------------------------------------------------
# Remote-proxy functions (call the Modal API endpoints)
# ------------------------------------------------------------------
def initialize_model():
    """
    Call the remote /initialize endpoint to load the model on the server.
    Matches the same call name used in your original UI.
    """
    try:
        resp = requests.post(f"{SERVER_URL}/initialize", json={"model_path": "Zyphra/Zonos-v0.1-transformer"}, timeout=120)
        if resp.status_code == 200:
            return "✅ Model loaded successfully on server!"
        else:
            return f"❌ Error loading model: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"❌ Exception calling server initialize: {str(e)}"

def create_speaker_embedding(wav_path: str) -> str:
    """
    Upload speaker audio file to /embedding endpoint and return embedding_b64 string.
    """
    if not wav_path:
        raise Exception("No speaker file provided")
    try:
        with open(wav_path, "rb") as fh:
            files = {"file": (os.path.basename(wav_path), fh, "application/octet-stream")}
            r = requests.post(f"{SERVER_URL}/embedding", files=files, timeout=120)
        if r.status_code != 200:
            raise Exception(f"Embedding failed: {r.status_code} {r.text}")
        data = r.json()
        return data["embedding_b64"]
    except Exception as e:
        raise Exception(f"create_speaker_embedding error: {str(e)}")

def _decode_wav_bytes_to_numpy(wav_bytes: bytes):
    """
    Return numpy array shaped (channels, samples), dtype=float32.
    """
    buf = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buf, dtype="float32")
    # sf.read returns (samples,) or (samples, channels)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T  # (channels, samples)
    return audio, sr

def stream_speech_generation(speaker_file, text, language, progress=gr.Progress()):
    """
    Stream speech generation using server SSE stream.
    This function preserves the original generator loop shape and yields (sr, audio_chunk)
    exactly as your original code expected.
    """
    if not speaker_file:
        gr.Warning("Please upload a speaker reference audio file.")
        return
    if not text or not text.strip():
        gr.Warning("Please enter some text to synthesize.")
        return

    try:
        progress(0.05, desc="Uploading speaker file and requesting embedding...")
        embedding_b64 = create_speaker_embedding(speaker_file)

        progress(0.25, desc="Starting synthesis stream from server...")
        payload = {"embedding_b64": embedding_b64, "text": text, "language": language}

        # Stream the server-sent events (SSE)
        with requests.post(f"{SERVER_URL}/synthesize_stream", json=payload, stream=True, timeout=3600) as r:
            if r.status_code != 200:
                gr.Error(f"Server stream error: {r.status_code} {r.text}")
                return

            chunk_count = 0
            total_samples = 0
            sample_rate = None

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # SSE line format: "data: <json>"
                if line.startswith("data: "):
                    payload_line = line[len("data: "):]
                    if payload_line == "__end__":
                        break
                    obj = json.loads(payload_line)
                    sr = int(obj.get("sr", 22050))
                    wav_b64 = obj["audio_b64"]
                    wav_bytes = base64.b64decode(wav_b64)
                    audio_np, sr2 = _decode_wav_bytes_to_numpy(wav_bytes)

                    # Ensure proper sampling rate variable
                    sample_rate = sr

                    chunk_count += 1
                    total_samples += audio_np.shape[-1]
                    duration_so_far = total_samples / sample_rate if sample_rate else 0.0

                    # Periodic progress updates
                    if chunk_count % 3 == 0:
                        progress(min(0.3 + (chunk_count * 0.02), 0.95),
                                 desc=f"Streaming: {duration_so_far:.1f}s ({chunk_count} chunks)")

                    # Yield exactly (sample_rate, audio_chunk) as in your original app
                    yield (sample_rate, audio_np)

            # Final progress update
            if sample_rate:
                final_duration = total_samples / sample_rate
                progress(1.0, desc=f"Streaming complete! {final_duration:.1f}s ({chunk_count} chunks)")

    except Exception as e:
        gr.Error(f"Synthesis error: {str(e)}")
        return

def process_audio_complete(speaker_file, text, language):
    """
    Request the full generated audio from the server (/synthesize_complete),
    return (sample_rate, numpy_audio) and status message — same signature as your earlier app.
    """
    if not speaker_file:
        return None, "❌ Please upload a speaker reference audio file."
    if not text or not text.strip():
        return None, "❌ Please enter some text to synthesize."

    try:
        embedding_b64 = create_speaker_embedding(speaker_file)
        payload = {"embedding_b64": embedding_b64, "text": text, "language": language}
        r = requests.post(f"{SERVER_URL}/synthesize_complete", json=payload, timeout=600)
        if r.status_code != 200:
            return None, f"❌ Server error: {r.status_code} {r.text}"

        wav_bytes = r.content
        audio_np, sr = _decode_wav_bytes_to_numpy(wav_bytes)

        # Flatten to 1D samples for Gradio Audio component if desired (keep as channels x samples)
        if audio_np.shape[0] > 1:
            # convert to interleaved mono (average) if you prefer single-channel:
            final = audio_np.flatten()
        else:
            final = audio_np.flatten()

        duration = final.shape[-1] / sr
        return (sr, final), f"✅ Complete! {duration:.1f}s audio ({audio_np.shape[0]} channels)"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def clear_outputs():
    """Clear outputs (same as original)."""
    return None, ""

def stop_streaming():
    """Stop current streaming (WebRTC handles stop)."""
    return "🛑 Streaming stopped (use stop button on WebRTC component)"

# ------------------------------------------------------------------
# Gradio UI — kept identical to your original layout and wiring
# ------------------------------------------------------------------
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
        
        1. **Initialize Model**: Click "Initialize Model" to load the Zonos speech synthesis model on remote server
        2. **Upload Speaker Audio**: Upload a reference audio file of the target speaker (WAV, MP3, etc.)
        3. **Enter Text**: Type or paste the text you want to convert to speech
        4. **Select Language**: Choose the appropriate language
        5. **Choose Generation Mode**:
           - **🎵 Start Real-Time Stream**: Streams audio chunks in real-time using SSE proxied into WebRTC
           - **🎯 Generate Complete Audio**: Traditional mode - generates complete audio file
        
        ### Tips:
        - Use clear, high-quality speaker reference audio (3-10 seconds recommended)
        - For best results use the same sampling rate (server handles resampling automatically if needed)
        - GPU acceleration on the server improves latency
        """)
    
    # Event handlers — identical wiring & loop to your original
    init_btn.click(
        fn=initialize_model,
        outputs=init_status
    )
    
    # Streaming: keep same signature, generator yields to WebRTC component
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
    
    # Auto-initialize on startup (calls remote initialize)
    demo.load(
        fn=initialize_model,
        outputs=init_status
    )

if __name__ == "__main__":
    # Keep ports and launch args similar to your original
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        quiet=False
    )
