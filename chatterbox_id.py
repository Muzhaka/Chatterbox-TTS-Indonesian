"""
Chatterbox TTS - Bahasa Indonesia + Voice Cloning
Jalan di GPU GTX 1060 6GB (CUDA 11.8 / PyTorch 2.5.1+cu118)

Cara pakai:
  source venv/Scripts/activate
  python chatterbox_id.py                          # TTS biasa (default text)
  python chatterbox_id.py "teks custom"            # TTS dengan teks sendiri
  python chatterbox_id.py "teks" ref.wav           # Voice cloning dari ref.wav
"""
import sys
import torch
import soundfile as sf
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
CHECKPOINT_FILENAME = "t3_cfg.safetensors"


def load_model(device="cuda"):
    print(f"[1/3] Load base model ke {device} ...")
    model = ChatterboxTTS.from_pretrained(device=device)

    print("[2/3] Download & load checkpoint Indonesian ...")
    ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILENAME)
    t3_state = load_file(ckpt_path, device="cpu")
    model.t3.load_state_dict(t3_state)
    if device == "cuda":
        torch.cuda.empty_cache()
    return model


def synthesize(model, text, out_path="output.wav", ref_audio=None):
    print(f"[3/3] Generate audio (voice cloning: {ref_audio or 'TIDAK'}) ...")
    if ref_audio:
        wav = model.generate(text, audio_prompt_path=ref_audio)
    else:
        wav = model.generate(text)
    sf.write(out_path, wav.squeeze().cpu().numpy(), model.sr)
    print(f"SELESAI -> {out_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | "
          f"GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

    text = sys.argv[1] if len(sys.argv) > 1 else (
        "Halo, ini contoh suara Bahasa Indonesia yang dihasilkan "
        "menggunakan Chatterbox TTS di komputer saya."
    )
    ref_audio = sys.argv[2] if len(sys.argv) > 2 else None
    out_path = sys.argv[3] if len(sys.argv) > 3 else "output.wav"

    model = load_model(device)
    synthesize(model, text, out_path, ref_audio)
