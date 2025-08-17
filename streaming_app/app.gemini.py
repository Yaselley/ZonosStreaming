# gradio_app.py
import gradio as gr
import modal
import numpy as np
import time
from gradio_webrtc import WebRTC

# --- Configuration ---
# The name of the Modal stub defined in modal_app.py
MODAL_APP_NAME = "zonos-tts-app"
# The name of the class in the Modal app
MODAL_CLASS_NAME = "ZonosSpeechSynthesizer"

def get_modal_function_handle():
    """Helper function to connect to the running Modal app."""
    try:
        return modal.Cls.lookup(MODAL_APP_NAME, MODAL_CLASS_NAME)
    except Exception:
        return None

def check_modal_status():
    """Checks if the Modal endpoint is ready."""
    handle = get_modal_function_handle()
    if handle:
        return "✅ Modal endpoint is ready."
    else:
        return "❌ Modal endpoint not found. Please deploy it first by running: `modal deploy modal_app.py`"

def read_audio_file(file_path):
    """Reads an audio file and returns its content as bytes."""
    if not file_path:
        return None
    with open(file_path, "rb") as f:
        return f.read()

def stream_speech_generation(speaker_file, text, language, progress=gr.Progress()):
    """
    Client-side function to handle real-time streaming.
    It calls the remote Modal generator and yields results to the Gradio WebRTC component.
    """
    if not speaker_file:
        gr.Warning("Please upload a speaker reference audio file.")
        return
    if not text.strip():
        gr.Warning("Please enter some text to synthesize.")
        return

    modal_handle = get_modal_function_handle()
    if not modal_handle:
        gr.Error("Modal endpoint is not available. Cannot synthesize.")
        return

    try:
        # Step 1: Read audio file into bytes
        progress(0.1, desc="Reading speaker audio...")
        audio_bytes = read_audio_file(speaker_file)
        if not audio_bytes:
            gr.Error("Could not read the speaker file.")
            return

        # Step 2: Call the remote method to create speaker embedding
        progress(0.2, desc="Creating speaker embedding on remote server...")
        zonos = modal_handle()
        speaker_embedding = zonos.create_speaker_embedding.remote(audio_bytes)

        # Step 3: Stream the synthesis
        progress(0.3, desc="Starting real-time synthesis stream...")
        chunk_count = 0
        total_samples = 0
        start_time = time.time()

        # Call the remote generator and iterate through the yielded chunks
        for sample_rate, audio_chunk_bytes in zonos.synthesize_stream.remote_gen(
            speaker_embedding, text, language
        ):
            # Convert the received bytes back to a numpy array
            audio_chunk = np.frombuffer(audio_chunk_bytes, dtype=np.float32)
            
            chunk_count += 1
            total_samples += len(audio_chunk)
            duration_so_far = total_samples / sample_rate
            
            # Update progress periodically
            if chunk_count % 3 == 0:
                progress(
                    min(0.3 + (chunk_count * 0.02), 0.95),
                    desc=f"Streaming: {duration_so_far:.1f}s ({chunk_count} chunks)",
                )
            
            # Yield the audio chunk to the WebRTC component for playback
            yield (sample_rate, audio_chunk)

        # Final progress update
        end_time = time.time()
        final_duration = total_samples / sample_rate
        processing_time = end_time - start_time
        progress(
            1.0,
            desc=f"Stream complete! {final_duration:.1f}s audio generated in {processing_time:.1f}s.",
        )

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return

# --- Gradio UI Definition ---
with gr.Blocks(
    title="🎙️ Real-Time Speech Synthesis with Modal",
    theme=gr.themes.Soft(),
) as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🎙️ Zonos Speech Synthesis (Powered by Modal)</h1>
        <p>Convert text to speech with custom voice cloning and <strong>real-time streaming</strong> via a serverless GPU backend.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Backend Status")
            init_status = gr.Textbox(
                label="Modal Endpoint Status",
                value="Checking status...",
                interactive=False,
            )
            gr.Markdown("### ⚙️ Input Configuration")
            speaker_file = gr.Audio(
                label="Speaker Reference Audio (3-10s recommended)",
                type="filepath",
                sources=["upload"],
            )
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text here...",
                lines=4,
                value="Hello! This is a test of a real-time speech synthesis system, powered by a serverless GPU.",
            )
            language_select = gr.Dropdown(
                choices=[
                    ("English (US)", "en-us"),
                    ("Spanish", "es"),
                    ("Italian", "it"),
                    ("Japanese", "ja"),
                ],
                label="Language",
                value="en-us",
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 🔊 Real-Time Audio Stream")
            streaming_audio = WebRTC(
                label="Live Audio Stream",
                mode="receive",
                modality="audio",
            )
            with gr.Row():
                stream_btn = gr.Button("🎵 Start Real-Time Stream", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

    # Event Handlers
    # When the "Start Stream" button is clicked, call the streaming function
    stream_btn.click(
        fn=stream_speech_generation,
        inputs=[speaker_file, text_input, language_select],
        outputs=[streaming_audio],
    )
    
    # The clear button stops the WebRTC component
    clear_btn.click(lambda: None, None, streaming_audio, queue=False)

    # Check Modal status when the Gradio app loads
    demo.load(fn=check_modal_status, outputs=init_status)

if __name__ == "__main__":
    print("Gradio app is starting...")
    print("Ensure the Modal app is deployed (`modal deploy modal_app.py`).")
    demo.launch(show_error=True, share=True)
