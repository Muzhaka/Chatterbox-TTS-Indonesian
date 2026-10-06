# Chatterbox TTS Indonesian - Setup Script
# Jalankan: python setup.py  (atau double-click ChatterboxTTS.bat)
# Script ini otomatis mengunduh model weights dari grandhigh/Chatterbox-TTS-Indonesian
# Model tidak disertakan di repo (5.32 GB) agar repo tetap ringan.

import os
import sys
import json
import shutil
from pathlib import Path

MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
CHECKPOINT_FILE = "t3_cfg.safetensors"
MODEL_DIR = Path(__file__).parent / "model_cache"
MODEL_PATH = MODEL_DIR / CHECKPOINT_FILE

def ensure_model():
    """Download model weights from Hugging Face if not already present."""
    if MODEL_PATH.exists():
        print(f"[setup] Model already cached: {MODEL_PATH}")
        return True

    print(f"[setup] Downloading model weights from {MODEL_REPO} ...")
    print(f"[setup] File: {CHECKPOINT_FILE} (~5.3 GB)")
    print(f"[setup] This may take several minutes depending on your connection.")

    try:
        from huggingface_hub import hf_hub_download
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=CHECKPOINT_FILE,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"[setup] Model downloaded: {path}")
        return True
    except Exception as e:
        print(f"[setup] ERROR downloading model: {e}")
        print(f"[setup] You can also download manually from:")
        print(f"[setup]   https://huggingface.co/{MODEL_REPO}/tree/main")
        print(f"[setup]   Save as: {MODEL_PATH}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("  Chatterbox TTS Indonesian - Setup")
    print("=" * 50)
    ok = ensure_model()
    if ok:
        print("[setup] Setup complete! You can now run ChatterboxTTS.bat")
    else:
        print("[setup] Setup incomplete. Please download model manually.")
    sys.exit(0 if ok else 1)