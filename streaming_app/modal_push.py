# modal_push.py
import os
import io
import json
import base64
import tempfile
from typing import Optional, Generator

import modal
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse

import torch
import torchaudio
import numpy as np

# --------------------------
# Modal app + image + volume
# --------------------------
app = modal.App("zonos-tts-app")

image = (
    modal.Image.debian_slim()
    .apt_install([
        "git",                # <--- ADDED: required for pip install from git
        "ca-certificates",    # <--- ADDED: ensures HTTPS certificate trust when cloning
        "espeak-ng",
        "ffmpeg",
        "libsndfile1",
        "libsox-dev",
        "build-essential"
    ])
    .pip_install("git+https://github.com/Yaselley/ZonosStreaming.git")  # Install Zonos from GitHub
    .pip_install([
        "gradio",
        "gradio_webrtc",
        "pydub"
    ])
    # Additional runtime packages required for model & inference:
    .pip_install([
        "torch",
        "torchaudio",
        "numpy",
        # If your ZonosStreaming repo provides the zonos package, you may not need to pip install zonos separately.
        # "zonos"
    ])
)

# Persistent cache volume for model files / huggingface cache (recommended)
cache_volume = modal.Volume.from_name("zonos-model-cache-vol", create_if_missing=True)

# --------------------------
# Shared model state (lazy)
# --------------------------
# We'll use module-level globals that are lazily initialized inside each function.
# This keeps the model resident in the container across requests.
MODEL = None
DEVICE = None
SAMPLING_RATE = None

def lazy_load_model():
    global MODEL, DEVICE, SAMPLING_RATE
    if MODEL is not None:
        return MODEL, DEVICE, SAMPLING_RATE

    # Import Zonos from the package installed by your git URL (or from pip if available)
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict  # type: ignore

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[modal] Loading Zonos model on device: {DEVICE}")
    MODEL = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEVICE)

    if DEVICE == "cuda":
        MODEL = MODEL.to(DEVICE, dtype=torch.bfloat16)
    else:
        MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    SAMPLING_RATE = getattr(MODEL.autoencoder, "sampling_rate", 22050)
    print("[modal] Model loaded; sampling_rate =", SAMPLING_RATE)
    return MODEL, DEVICE, SAMPLING_RATE

# Utility: save numpy/torch audio to WAV bytes and return a "data:audio/wav;base64,..." string
def wav_bytes_to_data_uri(audio_arr: np.ndarray, sample_rate: int) -> str:
    # Ensure float32 numpy
    if audio_arr.dtype != np.float32:
        audio_arr = audio_arr.astype(np.float32)

    # Convert to torch tensor (channels, samples)
    if audio_arr.ndim == 1:
        tensor = torch.from_numpy(audio_arr).unsqueeze(0)
    elif audio_arr.ndim == 2:
        tensor = torch.from_numpy(audio_arr)
    else:
        tensor = torch.from_numpy(audio_arr.squeeze())

    buffer = io.BytesIO()
    torchaudio.save(buffer, tensor, sample_rate, format="wav")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"

# Helper: create speaker embedding from uploaded file (UploadFile or base64 string)
def create_speaker_embedding_from_upload(upload_file: Optional[UploadFile], base64_str: Optional[str]):
    MODEL, DEVICE, SAMPLING_RATE = lazy_load_model()

    if upload_file is None and base64_str is None:
        raise ValueError("Provide speaker_file or speaker_base64")

    tmp_path = None
    try:
        if upload_file is not None:
            suffix = os.path.splitext(upload_file.filename)[1] or ".wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(upload_file.file.read())
            tmp.flush()
            tmp_path = tmp.name
        else:
            # base64 string may be a data URI or plain base64
            payload = base64_str.split(",")[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(base64.b64decode(payload))
            tmp.flush()
            tmp_path = tmp.name

        wav, sr = torchaudio.load(tmp_path)  # (channels, samples)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(DEVICE)
        speaker_embedding = MODEL.make_speaker_embedding(wav, sr)
        return speaker_embedding
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# --------------------------
# Non-streaming endpoint
# --------------------------
@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol/cache": cache_volume},
    timeout=3600,
    scaledown_window=600,
    max_containers=2,
)
@modal.fastapi_endpoint()
def synthesize(
    text: str = Form(...),
    language: str = Form("en-us"),
    speaker_file: UploadFile = File(None),
    speaker_base64: str = Form(None),
):
    """
    Non-streaming synthesis endpoint.
    Accepts multipart form: text (required), language (optional), speaker_file (upload) OR speaker_base64 (data URI).
    Returns JSON: {"audio_base64": "<base64 wav>", "sampling_rate": <int>}
    """
    try:
        MODEL, DEVICE, SAMPLING_RATE = lazy_load_model()

        # create speaker embedding
        speaker_embedding = create_speaker_embedding_from_upload(speaker_file, speaker_base64)

        # prepare conditioning & generate synchronously (collect chunks)
        from zonos.conditioning import make_cond_dict  # noqa: E402

        cond_dict = make_cond_dict(text=text, speaker=speaker_embedding, language=language)
        conditioning = MODEL.prepare_conditioning(cond_dict)

        audio_chunks = []
        for chunk in MODEL.stream(conditioning):
            if isinstance(chunk, torch.Tensor):
                arr = chunk.detach().cpu().numpy()
                if arr.ndim > 1:
                    arr = arr.squeeze()
            else:
                arr = np.asarray(chunk)
            audio_chunks.append(arr)

        if not audio_chunks:
            return JSONResponse({"error": "no audio generated"}, status_code=500)

        final_audio = np.concatenate(audio_chunks)
        data_uri = wav_bytes_to_data_uri(final_audio, SAMPLING_RATE)
        return JSONResponse({"audio_base64": data_uri, "sampling_rate": SAMPLING_RATE})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --------------------------
# Streaming endpoint
# --------------------------
@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol/cache": cache_volume},
    timeout=3600,
    scaledown_window=600,
    max_containers=1,
)
@modal.fastapi_endpoint()
def stream_synthesize(
    text: str = Form(...),
    language: str = Form("en-us"),
    speaker_file: UploadFile = File(None),
    speaker_base64: str = Form(None),
):
    """
    Streaming endpoint.
    Yields newline-delimited JSON strings. First yielded line is "START_STREAM".
    Each subsequent line is JSON for a chunk:
      {"action":"play_chunk", "audio": "data:audio/wav;base64,...", "chunk": n, "duration": seconds}
    Final line: {"action":"end_stream", "total_chunks": n, "total_duration": s}
    """

    def generator() -> Generator[bytes, None, None]:
        try:
            MODEL, DEVICE, SAMPLING_RATE = lazy_load_model()

            # create speaker embedding
            speaker_embedding = create_speaker_embedding_from_upload(speaker_file, speaker_base64)

            from zonos.conditioning import make_cond_dict  # noqa: E402

            cond_dict = make_cond_dict(text=text, speaker=speaker_embedding, language=language)
            conditioning = MODEL.prepare_conditioning(cond_dict)

            # Send a START_STREAM sentinel (matches your browser code)
            yield (f"START_STREAM\n").encode("utf-8")

            chunk_count = 0
            total_samples = 0

            for chunk in MODEL.stream(conditioning):
                chunk_count += 1

                if isinstance(chunk, torch.Tensor):
                    arr = chunk.detach().cpu().numpy()
                    if arr.ndim > 1:
                        arr = arr.squeeze()
                else:
                    arr = np.asarray(chunk)

                total_samples += arr.shape[-1]
                duration_so_far = float(total_samples) / float(SAMPLING_RATE)

                # Convert chunk to data URI base64 wav
                data_uri = wav_bytes_to_data_uri(arr, SAMPLING_RATE)

                stream_data = {
                    "action": "play_chunk",
                    "audio": data_uri,
                    "chunk": chunk_count,
                    "duration": duration_so_far,
                }

                # Yield JSON line + newline (client should parse each line)
                yield (json.dumps(stream_data) + "\n").encode("utf-8")

            # End-of-stream
            final_duration = float(total_samples) / float(SAMPLING_RATE)
            end_data = {
                "action": "end_stream",
                "total_chunks": chunk_count,
                "total_duration": final_duration,
            }
            yield (json.dumps(end_data) + "\n").encode("utf-8")

        except Exception as e:
            err = {"action": "error", "message": str(e)}
            yield (json.dumps(err) + "\n").encode("utf-8")

    # Use text/plain so the client gets the exact strings as they arrive
    return StreamingResponse(generator(), media_type="text/plain")

# --------------------------
# If run locally (not necessary for modal serve), provide a small message
# --------------------------
if __name__ == "__main__":
    print("This module is meant to be used with `modal serve modal_push.py` or `modal deploy modal_push.py`.")
