# modal_app.py
import modal
import torch
import torchaudio
from typing import Generator, Tuple

# Define the Modal application
app = modal.App("zonos-tts-app")

# Define the Docker image with all necessary dependencies
# This includes system packages and Python libraries.
# The Zonos library is installed directly from its GitHub repository.
zonos_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "ca-certificates",
        "espeak-ng",
        "ffmpeg",
        "libsndfile1",
        "libsox-dev",
        "build-essential",
    )
    .pip_install(
        "git+https://github.com/Yaselley/ZonosStreaming.git",
        "gradio",
        "torchaudio",
        "numpy",
    )
)

@app.cls(
    gpu="L4",  # Request an L4 GPU for faster inference
    image=zonos_image,
    container_idle_timeout=300,  # Shut down the container after 5 minutes of inactivity
)
class ZonosSpeechSynthesizer:
    """
    A class to handle the Zonos TTS model, managed by Modal.
    The model is loaded once when the container starts up.
    """

    def __init__(self):
        """
        The constructor is called once when the Modal container starts.
        It loads the Zonos model and prepares it for inference.
        """
        from zonos.model import Zonos

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading Zonos model...")
        # Load the pre-trained Zonos model
        self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer")
        
        # Move model to the appropriate device and set precision
        if self.device == "cuda":
            self.model = self.model.to(self.device, dtype=torch.bfloat16)
        else:
            self.model = self.model.to(self.device)
            
        self.model.eval()
        self.sampling_rate = getattr(self.model.autoencoder, "sampling_rate", 22050)
        print("✅ Model loaded successfully!")

    @modal.method()
    def create_speaker_embedding(self, audio_bytes: bytes):
        """
        Creates a speaker embedding from raw audio bytes.
        This is a remote method that can be called by the Gradio client.
        """
        import io

        print("Creating speaker embedding...")
        try:
            # Load audio from in-memory bytes
            wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
            wav = wav.to(self.device)
            
            # Ensure audio is mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
                
            embedding = self.model.make_speaker_embedding(wav, sr)
            print("✅ Speaker embedding created.")
            return embedding
        except Exception as e:
            print(f"❌ Error creating speaker embedding: {e}")
            raise

    @modal.method()
    def synthesize_stream(
        self, speaker_embedding, text: str, language: str
    ) -> Generator[Tuple[int, bytes], None, None]:
        """
        Generates streaming audio from text using the provided speaker embedding.
        This is a generator method that yields audio chunks as bytes.
        """
        from zonos.conditioning import make_cond_dict

        print(f"Starting synthesis for text: '{text[:30]}...'")
        try:
            # Prepare conditioning dictionary for the model
            cond_dict = make_cond_dict(
                text=text, speaker=speaker_embedding, language=language
            )
            conditioning = self.model.prepare_conditioning(cond_dict)

            # Stream the audio generation
            for i, audio_chunk in enumerate(self.model.stream(conditioning)):
                audio_np = audio_chunk.cpu().numpy()
                
                # Ensure correct shape and convert to bytes
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
                
                # Yield the sampling rate and the raw audio chunk as bytes
                yield (self.sampling_rate, audio_np.tobytes())
                print(f"   ... yielded chunk {i+1}")

            print("✅ Streaming finished.")
        except Exception as e:
            print(f"❌ Error during synthesis: {e}")
            raise

@app.local_entrypoint()
def main():
    """
    This function can be used to run the Modal app in serving mode.
    Deploy with `modal deploy modal_app.py`.
    """
    print("To deploy this app, run: modal deploy modal_app.py")
