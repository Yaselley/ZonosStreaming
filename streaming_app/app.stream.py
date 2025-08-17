import torch
import torchaudio
import gradio as gr
import io
import numpy as np
from typing import Generator, Tuple
import tempfile
import os
import base64
import json
import time

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

    def synthesize_stream(self, speaker_embedding, text: str, language: str) -> Generator[np.ndarray, None, None]:
        """Generate streaming audio from text"""
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
                    yield audio_np
                else:
                    yield audio_chunk
                    
        except Exception as e:
            raise Exception(f"Error during synthesis: {str(e)}")

def audio_to_base64_wav(audio_data, sample_rate):
    """Convert audio numpy array to base64 encoded WAV"""
    # Create WAV file in memory
    buffer = io.BytesIO()
    
    # Ensure audio is in the right format
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Reshape for torchaudio (channels, samples)
    if audio_data.ndim == 1:
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
    else:
        audio_tensor = torch.from_numpy(audio_data)
    
    # Save as WAV
    torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
    
    # Get base64 string
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode()
    
    return f"data:audio/wav;base64,{audio_base64}"

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

def process_audio_real_streaming(speaker_file, text, language, progress=gr.Progress()):
    """Process text-to-speech with real-time streaming playback"""
    global synthesizer
    
    if synthesizer is None:
        yield "", "❌ Model not initialized. Please initialize the model first."
        return
    
    if not speaker_file:
        yield "", "❌ Please upload a speaker reference audio file."
        return
    
    if not text.strip():
        yield "", "❌ Please enter some text to synthesize."
        return
    
    try:
        progress(0.1, desc="Creating speaker embedding...")
        
        # Create speaker embedding
        speaker_embedding = synthesizer.create_speaker_embedding(speaker_file)
        
        progress(0.3, desc="Starting real-time synthesis...")
        
        # Initialize streaming
        chunk_count = 0
        total_samples = 0
        
        # Signal start of streaming
        yield "START_STREAM", "🎵 Starting real-time playback..."
        
        for audio_chunk in synthesizer.synthesize_stream(speaker_embedding, text, language):
            chunk_count += 1
            total_samples += len(audio_chunk)
            duration_so_far = total_samples / synthesizer.sampling_rate
            
            # Convert chunk to base64 WAV for immediate playback
            audio_b64 = audio_to_base64_wav(audio_chunk, synthesizer.sampling_rate)
            
            # Create streaming command
            stream_data = {
                "action": "play_chunk",
                "audio": audio_b64,
                "chunk": chunk_count,
                "duration": duration_so_far
            }
            
            # Update progress
            if chunk_count % 3 == 0:
                progress(
                    min(0.3 + (chunk_count * 0.02), 0.95), 
                    desc=f"Streaming: {duration_so_far:.1f}s"
                )
            
            yield json.dumps(stream_data), f"🎵 Playing: {duration_so_far:.1f}s ({chunk_count} chunks)"
            
            # Small delay to prevent overwhelming the browser
            time.sleep(0.01)
        
        # Signal end of streaming
        final_duration = total_samples / synthesizer.sampling_rate
        end_data = {
            "action": "end_stream",
            "total_chunks": chunk_count,
            "total_duration": final_duration
        }
        
        progress(1.0, desc="Streaming complete!")
        yield json.dumps(end_data), f"✅ Streaming complete! {final_duration:.1f}s ({chunk_count} chunks)"
            
    except Exception as e:
        error_data = {"action": "error", "message": str(e)}
        yield json.dumps(error_data), f"❌ Error: {str(e)}"

def process_audio_non_streaming(speaker_file, text, language):
    """Process text-to-speech without streaming (alternative approach)"""
    global synthesizer
    
    if synthesizer is None:
        return "", None, "❌ Model not initialized. Please initialize the model first."
    
    if not speaker_file:
        return "", None, "❌ Please upload a speaker reference audio file."
    
    if not text.strip():
        return "", None, "❌ Please enter some text to synthesize."
    
    try:
        # Create speaker embedding
        speaker_embedding = synthesizer.create_speaker_embedding(speaker_file)
        
        # Collect all audio chunks first
        audio_chunks = []
        for audio_chunk in synthesizer.synthesize_stream(speaker_embedding, text, language):
            audio_chunks.append(audio_chunk)
        
        # Return final result
        if audio_chunks:
            final_audio = np.concatenate(audio_chunks)
            final_duration = len(final_audio) / synthesizer.sampling_rate
            return "", (synthesizer.sampling_rate, final_audio), f"✅ Complete! {final_duration:.1f}s audio ({len(audio_chunks)} chunks)"
        else:
            return "", None, "❌ No audio generated"
            
    except Exception as e:
        return "", None, f"❌ Error: {str(e)}"

def clear_outputs():
    """Clear all outputs"""
    return "", "", None, ""

def stop_streaming():
    """Stop current streaming"""
    return json.dumps({"action": "stop"}), "🛑 Streaming stopped"

# Create Gradio interface
with gr.Blocks(
    title="🎙️ Zonos Real-Time Speech Synthesis",
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
    #audio-player {
        width: 100%;
        margin: 1rem 0;
    }
    .streaming-controls {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    """,
    head="""
    <script>
    let audioContext = null;
    let audioQueue = [];
    let isPlaying = false;
    let currentSource = null;

    async function initAudioContext() {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
    }

    async function playAudioChunk(audioBase64) {
        try {
            await initAudioContext();
            
            // Convert base64 to array buffer
            const audioData = atob(audioBase64.split(',')[1]);
            const arrayBuffer = new ArrayBuffer(audioData.length);
            const view = new Uint8Array(arrayBuffer);
            for (let i = 0; i < audioData.length; i++) {
                view[i] = audioData.charCodeAt(i);
            }
            
            // Decode audio data
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Create and play source
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start();
            
            return audioBuffer.duration;
        } catch (error) {
            console.error('Error playing audio chunk:', error);
            return 0;
        }
    }

    function handleStreamData(data) {
        if (!data || data === "START_STREAM") return;
        
        try {
            const streamData = JSON.parse(data);
            
            switch (streamData.action) {
                case 'play_chunk':
                    playAudioChunk(streamData.audio);
                    break;
                case 'end_stream':
                    console.log('Streaming ended:', streamData.total_duration + 's');
                    break;
                case 'stop':
                    if (currentSource) {
                        currentSource.stop();
                        currentSource = null;
                    }
                    break;
                case 'error':
                    console.error('Streaming error:', streamData.message);
                    break;
            }
        } catch (error) {
            console.error('Error handling stream data:', error);
        }
    }

    // Auto-initialize audio context on user interaction
    document.addEventListener('click', initAudioContext, { once: true });
    </script>
    """
) as demo:
    
    gr.HTML("""
    <div class="main-header">
        <h1>🎙️ Zonos Real-Time Speech Synthesis</h1>
        <p>Convert text to speech with custom speaker voice cloning and <strong>real-time streaming playback</strong></p>
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
                max_lines=10
            )
            
            language_select = gr.Dropdown(
                choices=[
                    ("English (US)", "en-us"),
                    ("Spanish", "es"),
                    ("Italian", "it")
                ],
                label="Language",
                value="en-us"
            )
            
            with gr.Row():
                stream_btn = gr.Button("🎵 Real-Time Stream", variant="primary", size="lg")
                stop_btn = gr.Button("🛑 Stop Stream", variant="secondary")
            
            with gr.Row():
                generate_direct_btn = gr.Button("🎯 Generate Complete", variant="secondary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Output")
            
            # Hidden textbox for streaming data
            stream_data = gr.Textbox(
                visible=False,
                elem_id="stream-data"
            )
            
            # Regular audio for complete generation
            output_audio = gr.Audio(
                label="Generated Speech (Complete Mode)",
                interactive=False,
                autoplay=False
            )
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )
    
    gr.Markdown("---")
    
    gr.HTML("""
    <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4>🎧 Real-Time Streaming Audio Player</h4>
        <p><strong>Real-Time Stream mode</strong> plays audio chunks immediately as they're generated!</p>
        <p><em>Click anywhere on the page first to enable audio, then use "Real-Time Stream" button.</em></p>
    </div>
    """)
    
    with gr.Accordion("📖 Usage Instructions", open=False):
        gr.Markdown("""
        ### How to use Real-Time Streaming:
        
        1. **Initialize Model**: Click "Initialize Model" to load the Zonos speech synthesis model
        2. **Upload Speaker Audio**: Upload a reference audio file of the target speaker (WAV, MP3, etc.)
        3. **Enter Text**: Type or paste the text you want to convert to speech
        4. **Select Language**: Choose the appropriate language (English, Spanish, or Italian)
        5. **Click anywhere on the page** to enable audio context (browser requirement)
        6. **Choose Generation Mode**:
           - **🎵 Real-Time Stream**: Plays audio chunks immediately as they're generated
           - **🎯 Generate Complete**: Traditional mode - generates complete audio then plays
           - **🛑 Stop Stream**: Stops current streaming playback
        
        ### Real-Time Streaming Features:
        - ✅ **True real-time playback** - hear audio as it's being generated
        - ✅ **No waiting** - audio starts playing immediately
        - ✅ **Continuous stream** - seamless playback of chunks
        - ✅ **Live progress** - see generation progress in real-time
        - ✅ **Stop anytime** - interrupt generation if needed
        
        ### Tips:
        - **Click on the page first** - browsers require user interaction before audio playback
        - Use clear, high-quality speaker reference audio (3-10 seconds recommended)
        - Real-time streaming works best with shorter texts (< 200 words)
        - GPU acceleration significantly improves streaming performance
        - If audio doesn't play, check browser audio permissions
        """)
    
    # JavaScript to handle stream data
    stream_data.change(
        fn=None,
        inputs=stream_data,
        outputs=None,
        js="(data) => { handleStreamData(data); return data; }"
    )
    
    # Event handlers
    init_btn.click(
        fn=initialize_model,
        outputs=init_status
    )
    
    stream_btn.click(
        fn=process_audio_real_streaming,
        inputs=[speaker_file, text_input, language_select],
        outputs=[stream_data, status_output]
    )
    
    stop_btn.click(
        fn=stop_streaming,
        outputs=[stream_data, status_output]
    )
    
    generate_direct_btn.click(
        fn=process_audio_non_streaming,
        inputs=[speaker_file, text_input, language_select],
        outputs=[stream_data, output_audio, status_output]
    )
    
    clear_btn.click(
        fn=clear_outputs,
        outputs=[stream_data, stream_data, output_audio, status_output]
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