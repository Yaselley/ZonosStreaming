import torch
import torchaudio
import gradio as gr
import numpy as np
from typing import Generator, Tuple
import tempfile
import os

# Import Gradio WebRTC for streaming
from gradio_webrtc import WebRTC

# Import Zonos components
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

class ZonosSpeechSynthesizer:
    def __init__(self, model_path="Zyphra/Zonos-v0.1-transformer", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the model
        print("Loading Zonos model...")
        self.model = Zonos.from_pretrained(model_path, device=self.device)
        
        if self.device == "cuda":
            self.model = self.model.to(self.device, dtype=torch.bfloat16)
        else:
            self.model = self.model.to(self.device)
            
        self.sampling_rate = getattr(self.model.autoencoder, "sampling_rate", 22050)
        self.model.eval()
        print("Model loaded successfully!")

    def create_speaker_embedding(self, wav_path: str):
        """Create speaker embedding from audio file"""
        try:
            wav, sr = torchaudio.load(wav_path)
            wav = wav.to(self.device)
            
            # Ensure mono audio
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            return self.model.make_speaker_embedding(wav, sr)
        except Exception as e:
            raise Exception(f"Error creating speaker embedding: {str(e)}")

    def synthesize_stream(self, speaker_embedding, text: str, language: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generate streaming audio from text - yields (sample_rate, audio_chunk) tuples"""
        try:
            # Prepare conditioning
            cond_dict = make_cond_dict(
                text=text, 
                speaker=speaker_embedding, 
                language=language
            )
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # Stream generation
            for audio_chunk in self.model.stream(conditioning):
                # Convert to numpy for processing
                if isinstance(audio_chunk, torch.Tensor):
                    audio_np = audio_chunk.cpu().numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np.squeeze()
                else:
                    audio_np = audio_chunk
                
                # Ensure proper shape for gradio (channels, samples)
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
                
                yield (self.sampling_rate, audio_np)
                    
        except Exception as e:
            raise Exception(f"Error during synthesis: {str(e)}")

# Initialize the synthesizer
synthesizer = None

def initialize_model():
    """Initialize the model with error handling"""
    global synthesizer
    try:
        synthesizer = ZonosSpeechSynthesizer()
        return "✅ Model loaded successfully!"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def stream_speech_generation(speaker_file, text, language, progress=gr.Progress()):
    """Stream speech generation using WebRTC"""
    global synthesizer
    
    if synthesizer is None:
        # Try to initialize if not done
        init_result = initialize_model()
        if "❌" in init_result:
            return
    
    if not speaker_file:
        gr.Warning("Please upload a speaker reference audio file.")
        return
    
    if not text.strip():
        gr.Warning("Please enter some text to synthesize.")
        return
    
    try:
        progress(0.1, desc="Creating speaker embedding...")
        
        # Create speaker embedding
        speaker_embedding = synthesizer.create_speaker_embedding(speaker_file)
        
        progress(0.3, desc="Starting real-time synthesis streaming...")
        
        chunk_count = 0
        total_samples = 0
        
        # Stream audio chunks
        for sample_rate, audio_chunk in synthesizer.synthesize_stream(speaker_embedding, text, language):
            chunk_count += 1
            total_samples += audio_chunk.shape[-1]  # Last dimension is samples
            duration_so_far = total_samples / sample_rate
            
            # Update progress periodically
            if chunk_count % 3 == 0:
                progress(
                    min(0.3 + (chunk_count * 0.02), 0.95), 
                    desc=f"Streaming: {duration_so_far:.1f}s ({chunk_count} chunks)"
                )
            
            yield (sample_rate, audio_chunk)
        
        # Final progress update
        final_duration = total_samples / sample_rate
        progress(1.0, desc=f"Streaming complete! {final_duration:.1f}s ({chunk_count} chunks)")
            
    except Exception as e:
        gr.Error(f"Synthesis error: {str(e)}")
        return

def process_audio_complete(speaker_file, text, language):
    """Process text-to-speech without streaming (complete generation)"""
    global synthesizer
    
    if synthesizer is None:
        return None, "❌ Model not initialized. Please initialize the model first."
    
    if not speaker_file:
        return None, "❌ Please upload a speaker reference audio file."
    
    if not text.strip():
        return None, "❌ Please enter some text to synthesize."
    
    try:
        # Create speaker embedding
        speaker_embedding = synthesizer.create_speaker_embedding(speaker_file)
        
        # Collect all audio chunks first
        audio_chunks = []
        total_samples = 0
        sample_rate = None
        
        for sr, audio_chunk in synthesizer.synthesize_stream(speaker_embedding, text, language):
            sample_rate = sr
            # Convert from (channels, samples) to flat array
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.flatten()
            audio_chunks.append(audio_chunk)
            total_samples += len(audio_chunk)
        
        # Return final result
        if audio_chunks:
            final_audio = np.concatenate(audio_chunks)
            final_duration = total_samples / sample_rate
            return (sample_rate, final_audio), f"✅ Complete! {final_duration:.1f}s audio ({len(audio_chunks)} chunks)"
        else:
            return None, "❌ No audio generated"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def clear_outputs():
    """Clear all outputs"""
    return None, ""

def stop_streaming():
    """Stop current streaming"""
    # The WebRTC component handles stopping automatically
    return "🛑 Streaming stopped (use stop button on WebRTC component)"

# Create Gradio interface
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
        
        1. **Initialize Model**: Click "Initialize Model" to load the Zonos speech synthesis model
        2. **Upload Speaker Audio**: Upload a reference audio file of the target speaker (WAV, MP3, etc.)
        3. **Enter Text**: Type or paste the text you want to convert to speech
        4. **Select Language**: Choose the appropriate language (English, Spanish, Japanese, or Italian, and more ...)
        5. **Choose Generation Mode**:
           - **🎵 Start Real-Time Stream**: Streams audio chunks in real-time using WebRTC
           - **🎯 Generate Complete Audio**: Traditional mode - generates complete audio file
        
        ### WebRTC Streaming Features:
        - ✅ **True real-time streaming** - audio plays as it's generated
        - ✅ **Low latency** - WebRTC provides optimal audio streaming
        - ✅ **Built-in controls** - WebRTC component has play/pause/stop controls
        - ✅ **Automatic buffering** - Handles network variations smoothly
        - ✅ **No browser compatibility issues** - WebRTC is well-supported
        - ✅ **Professional quality** - Industry-standard streaming protocol
        
        ### WebRTC vs Custom JavaScript:
        - **Better performance**: WebRTC is optimized for real-time media
        - **More reliable**: Less prone to timing and buffering issues
        - **Cleaner code**: No custom audio context management needed
        - **Standard protocol**: Uses established WebRTC standards
        - **Built-in controls**: Player controls are handled automatically
        
        ### Tips:
        - Use clear, high-quality speaker reference audio (3-10 seconds recommended)
        - Real-time streaming works best with moderate text lengths (< 500 words)
        - GPU acceleration significantly improves streaming performance
        - The WebRTC component will show connection status automatically
        """)
    
    # Event handlers
    init_btn.click(
        fn=initialize_model,
        outputs=init_status
    )
    
    # WebRTC streaming
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
    
    # Auto-initialize on startup
    demo.load(
        fn=initialize_model,
        outputs=init_status
    )

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        quiet=False
    )