"""
Chatterbox TTS Indonesian - Desktop App (PyWebView)
Double-click ChatterboxTTS.bat untuk buka.
"""

import os
import sys
import json
import time
import base64
import shutil
import threading
import traceback
from pathlib import Path

import torch
import soundfile as sf
import webview

from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ── Config ──────────────────────────────────────
MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
CHECKPOINT_FILE = "t3_cfg.safetensors"

APP_DIR = Path(__file__).parent
PROFILE_MODEL_DIR = APP_DIR / "profile_model"
BASE_CLONE_DIR = APP_DIR / "base_clone"
TTS_OUTPUT_DIR = APP_DIR / "tts_outputs"
CLONE_OUTPUT_DIR = APP_DIR / "output_clone"
HTML_FILE = APP_DIR / "gui.html"
VOICE_PROFILES_FILE = APP_DIR / "voice_profiles.json"
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

DEFAULT_PARAMS = {
    "repetition_penalty": 1.2,
    "min_p": 0.05,
    "top_p": 1.0,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
}

for d in (PROFILE_MODEL_DIR, BASE_CLONE_DIR, TTS_OUTPUT_DIR, CLONE_OUTPUT_DIR):
    d.mkdir(exist_ok=True)


# ── Voice Profiles ──────────────────────────────
def load_voice_profiles():
    if VOICE_PROFILES_FILE.exists():
        try:
            data = json.loads(VOICE_PROFILES_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def save_voice_profiles(profiles):
    tmp = VOICE_PROFILES_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(VOICE_PROFILES_FILE)


def _profile_voice_path(pid):
    return PROFILE_MODEL_DIR / pid / "voice.wav"


def ensure_voice_profiles():
    profiles = load_voice_profiles()
    if not profiles:
        profiles = [
            {
                "id": "default",
                "name": "Default",
                "desc": "Suara bawaan Chatterbox",
                "params": dict(DEFAULT_PARAMS),
            }
        ]
        save_voice_profiles(profiles)
    return profiles


# ── Model ───────────────────────────────────────
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
_user_device = device
is_generating = False
_model_lock = threading.Lock()


def load_model_safe(target_device=None):
    global model, device, _user_device
    with _model_lock:
        if target_device is None:
            target_device = _user_device
        if target_device == "cuda" and not torch.cuda.is_available():
            target_device = "cpu"
        try:
            m = ChatterboxTTS.from_pretrained(device=target_device)
            ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE)
            t3_state = load_file(ckpt_path, device="cpu")
            m.t3.load_state_dict(t3_state)
            if target_device == "cuda":
                torch.cuda.empty_cache()
            model = m
            device = target_device
            _user_device = target_device
            return {"ok": True, "device": device, "gpu": gpu_name()}
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": str(e)}


def gpu_name():
    try:
        if device == "cuda":
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def reload_model_on_new_device():
    global model
    try:
        if model is not None and device == "cuda":
            del model
            torch.cuda.empty_cache()
    except Exception:
        pass
    model = None
    return load_model_safe()


def _sync_status():
    if model is None:
        return {"loaded": False, "device": device, "gpu": gpu_name()}
    return {"loaded": True, "device": device, "gpu": gpu_name()}


# ── File dialog helpers ─────────────────────────
_main_window = None


def _tk_open_dialog(title, patterns):
    box = {"path": None}

    def _run():
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update_idletasks()
            p = filedialog.askopenfilename(title=title, filetypes=patterns)
            try:
                root.destroy()
            except Exception:
                pass
            box["path"] = p or None
        except Exception as e:
            box["error"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(10)
    return box.get("path"), box.get("error")


def ask_audio_file():
    patterns = [
        ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"),
        ("WAV", "*.wav"),
        ("MP3", "*.mp3"),
        ("All Files", "*.*"),
    ]
    try:
        if _main_window is not None:
            res = _main_window.create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=False,
                file_types=(
                    "Audio Files (*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac)",
                    "All files (*.*)",
                ),
            )
            if res:
                path = res[0] if isinstance(res, (list, tuple)) else res
                if path:
                    return str(path), None
    except Exception:
        pass
    return _tk_open_dialog("Pilih file suara", patterns)


# ── Helpers ─────────────────────────────────────
def _data_url(path):
    try:
        p = Path(path)
        if not p.exists():
            return {"ok": False, "error": "File tidak ditemukan"}
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return {"ok": True, "data_url": f"data:audio/wav;base64,{b64}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── API ─────────────────────────────────────────
class Api:
    # ── Status / device ─────────────────────────
    def get_status(self):
        return _sync_status()

    def set_device(self, dev):
        global _user_device
        if dev not in ("cuda", "cpu"):
            return {"ok": False, "error": "Device tidak valid"}
        if dev == "cuda" and not torch.cuda.is_available():
            return {"ok": False, "error": "CUDA tidak tersedia"}
        _user_device = dev
        res = reload_model_on_new_device()
        if res.get("ok"):
            return {"ok": True, "device": res["device"], "gpu": res["gpu"]}
        return res

    def close_app(self):
        def _shutdown():
            time.sleep(0.15)
            try:
                if _main_window is not None:
                    _main_window.destroy()
            except Exception:
                pass
            os._exit(0)

        threading.Thread(target=_shutdown, daemon=True).start()
        return {"ok": True}

    # ── Voice Profiles ──────────────────────────
    def get_profiles(self):
        profiles = ensure_voice_profiles()
        out = []
        for p in profiles:
            pid = p.get("id", "default")
            voice = _profile_voice_path(pid)
            item = dict(p)
            item["has_voice"] = voice.exists()
            item["params"] = p.get("params") or dict(DEFAULT_PARAMS)
            out.append(item)
        return out

    def add_profile(self, name, desc):
        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "Nama profil kosong"}
        profiles = ensure_voice_profiles()
        pid = "voice_" + str(int(time.time() * 1000))
        entry = {
            "id": pid,
            "name": name,
            "desc": (desc or "").strip(),
            "params": dict(DEFAULT_PARAMS),
        }
        (PROFILE_MODEL_DIR / pid).mkdir(parents=True, exist_ok=True)
        profiles.append(entry)
        save_voice_profiles(profiles)
        return {"ok": True, "id": pid}

    def save_profile(self, pid, name, desc, params):
        profiles = ensure_voice_profiles()
        merged = dict(DEFAULT_PARAMS)
        if isinstance(params, dict):
            merged.update({k: v for k, v in params.items() if v is not None})
        found = False
        for p in profiles:
            if p.get("id") == pid:
                p["name"] = (name or p.get("name") or pid).strip()
                p["desc"] = (desc or "").strip()
                p["params"] = merged
                found = True
                break
        if not found:
            profiles.append(
                {
                    "id": pid,
                    "name": (name or pid).strip(),
                    "desc": (desc or "").strip(),
                    "params": merged,
                }
            )
        if pid != "default":
            (PROFILE_MODEL_DIR / pid).mkdir(parents=True, exist_ok=True)
        save_voice_profiles(profiles)
        return {"ok": True}

    def delete_profile(self, pid):
        if pid == "default":
            return {"ok": False, "error": "Profil default tidak bisa dihapus"}
        profiles = ensure_voice_profiles()
        profiles = [p for p in profiles if p.get("id") != pid]
        save_voice_profiles(profiles)
        folder = PROFILE_MODEL_DIR / pid
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        return {"ok": True}

    def pick_audio_file(self):
        path, err = ask_audio_file()
        if path:
            return {"ok": True, "path": path, "name": Path(path).name}
        if err:
            return {"ok": False, "error": err}
        return {"ok": False, "error": "Tidak ada file dipilih"}

    def upload_voice(self, pid, src_path):
        if pid == "default":
            return {"ok": False, "error": "Profil default tidak bisa diubah"}
        src = Path(src_path)
        if not src.exists() or not src.is_file():
            return {"ok": False, "error": "File tidak ditemukan"}
        if src.suffix.lower() not in AUDIO_EXTS:
            return {"ok": False, "error": "Format audio tidak didukung"}
        folder = PROFILE_MODEL_DIR / pid
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / "voice.wav"
        try:
            if src.suffix.lower() == ".wav":
                shutil.copy2(src, dst)
            else:
                import numpy as np
                data, sr = sf.read(str(src))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                sf.write(str(dst), np.asarray(data, dtype="float32"), sr)
        except Exception as e:
            return {"ok": False, "error": f"Gagal memproses audio: {e}"}
        return {"ok": True, "path": str(dst), "name": dst.name}

    def get_profile_voice_url(self, pid):
        voice = _profile_voice_path(pid)
        if not voice.exists():
            return {"ok": False, "error": "Profil ini belum punya file suara"}
        return _data_url(voice)

    # ── Clone source ────────────────────────────
    def set_clone_source(self, src_path):
        src = Path(src_path)
        if not src.exists() or not src.is_file():
            return {"ok": False, "error": "File tidak ditemukan"}
        if src.suffix.lower() not in AUDIO_EXTS:
            return {"ok": False, "error": "Format audio tidak didukung"}
        BASE_CLONE_DIR.mkdir(exist_ok=True)
        for old in BASE_CLONE_DIR.glob("*"):
            try:
                old.unlink()
            except Exception:
                pass
        dst = BASE_CLONE_DIR / ("source" + src.suffix.lower())
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            return {"ok": False, "error": f"Gagal menyalin file: {e}"}
        return {"ok": True, "name": dst.name, "path": str(dst)}

    def clone_source_url(self):
        files = [f for f in BASE_CLONE_DIR.glob("*") if f.suffix.lower() in AUDIO_EXTS]
        if not files:
            return {"ok": False, "error": "Belum ada file sumber"}
        return _data_url(files[0])

    # ── Generate ────────────────────────────────
    def generate(self, text, profile_id=None, params=None):
        global is_generating
        if is_generating:
            return {"ok": False, "error": "Masih ada proses lain, tunggu sebentar..."}
        if not text or not text.strip():
            return {"ok": False, "error": "Teks kosong"}
        if model is None:
            return {"ok": False, "error": "Model belum siap, tunggu sampai status Ready"}

        p_params = dict(DEFAULT_PARAMS)
        ref_path = None
        if profile_id and profile_id != "default":
            for p in ensure_voice_profiles():
                if p.get("id") == profile_id:
                    voice = _profile_voice_path(profile_id)
                    if voice.exists():
                        ref_path = str(voice)
                    p_params.update(p.get("params") or {})
                    break
        if isinstance(params, dict):
            p_params.update({k: v for k, v in params.items() if v is not None})

        is_generating = True
        try:
            TTS_OUTPUT_DIR.mkdir(exist_ok=True)
            idx = len(list(TTS_OUTPUT_DIR.glob("*.wav"))) + 1
            out_file = TTS_OUTPUT_DIR / f"output_{idx:04d}.wav"
            gen_kwargs = {
                "text": text.strip(),
                "repetition_penalty": float(p_params.get("repetition_penalty", 1.2)),
                "min_p": float(p_params.get("min_p", 0.05)),
                "top_p": float(p_params.get("top_p", 1.0)),
                "exaggeration": float(p_params.get("exaggeration", 0.5)),
                "cfg_weight": float(p_params.get("cfg_weight", 0.5)),
                "temperature": float(p_params.get("temperature", 0.8)),
            }
            if ref_path:
                gen_kwargs["audio_prompt_path"] = ref_path

            wav = model.generate(**gen_kwargs)
            wav_np = wav.squeeze().cpu().numpy()
            sf.write(str(out_file), wav_np, model.sr)

            res = _data_url(out_file)
            if not res.get("ok"):
                return res
            return {
                "ok": True,
                "file": str(out_file),
                "filename": out_file.name,
                "size_kb": round(out_file.stat().st_size / 1024, 1),
                "data_url": res["data_url"],
            }
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": str(e)}
        finally:
            is_generating = False

    def generate_clone(self, text, params=None):
        global is_generating
        if is_generating:
            return {"ok": False, "error": "Masih ada proses lain, tunggu sebentar..."}
        if not text or not text.strip():
            return {"ok": False, "error": "Teks kosong"}
        if model is None:
            return {"ok": False, "error": "Model belum siap, tunggu sampai status Ready"}

        files = [f for f in BASE_CLONE_DIR.glob("*") if f.suffix.lower() in AUDIO_EXTS]
        if not files:
            return {"ok": False, "error": "Belum ada file suara sumber untuk kloning"}
        ref_path = str(files[0])

        p_params = dict(DEFAULT_PARAMS)
        if isinstance(params, dict):
            p_params.update({k: v for k, v in params.items() if v is not None})

        is_generating = True
        try:
            CLONE_OUTPUT_DIR.mkdir(exist_ok=True)
            idx = len(list(CLONE_OUTPUT_DIR.glob("*.wav"))) + 1
            out_file = CLONE_OUTPUT_DIR / f"clone_{idx:04d}.wav"

            gen_kwargs = {
                "text": text.strip(),
                "audio_prompt_path": ref_path,
                "repetition_penalty": float(p_params.get("repetition_penalty", 1.2)),
                "min_p": float(p_params.get("min_p", 0.05)),
                "top_p": float(p_params.get("top_p", 1.0)),
                "exaggeration": float(p_params.get("exaggeration", 0.5)),
                "cfg_weight": float(p_params.get("cfg_weight", 0.5)),
                "temperature": float(p_params.get("temperature", 0.8)),
            }

            wav = model.generate(**gen_kwargs)
            wav_np = wav.squeeze().cpu().numpy()
            sf.write(str(out_file), wav_np, model.sr)

            for f in files:
                try:
                    f.unlink()
                except Exception:
                    pass

            res = _data_url(out_file)
            if not res.get("ok"):
                return res
            return {
                "ok": True,
                "file": str(out_file),
                "filename": out_file.name,
                "size_kb": round(out_file.stat().st_size / 1024, 1),
                "data_url": res["data_url"],
            }
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": str(e)}
        finally:
            is_generating = False

    def list_outputs(self):
        TTS_OUTPUT_DIR.mkdir(exist_ok=True)
        files = sorted(TTS_OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"name": f.name, "path": str(f), "size_kb": round(f.stat().st_size / 1024, 1)} for f in files[:20]]

    def list_clone_outputs(self):
        CLONE_OUTPUT_DIR.mkdir(exist_ok=True)
        files = sorted(CLONE_OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"name": f.name, "path": str(f), "size_kb": round(f.stat().st_size / 1024, 1)} for f in files[:20]]

    def delete_output(self, path, kind="generate"):
        try:
            p = Path(path)
            target_dir = CLONE_OUTPUT_DIR if kind == "clone" else TTS_OUTPUT_DIR
            if p.exists() and p.parent == target_dir:
                p.unlink()
                return {"ok": True}
            return {"ok": False, "error": "File tidak ditemukan"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_audio_data_url(self, path):
        return _data_url(path)

    def open_folder(self, kind="generate"):
        target_dir = CLONE_OUTPUT_DIR if kind == "clone" else TTS_OUTPUT_DIR
        target_dir.mkdir(exist_ok=True)
        os.startfile(str(target_dir))
        return {"ok": True}

    # ── File Picker modal ────────────────────────
    def pick_profile_voice(self):
        """Modal untuk pilih file voice untuk profile."""
        path, err = ask_audio_file()
        if path:
            return {"ok": True, "path": path, "name": Path(path).name}
        if err:
            return {"ok": False, "error": err}
        return {"ok": False, "error": "Tidak ada file dipilih"}


# ── Main ────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Chatterbox TTS Indonesian - Desktop App")
    print("=" * 50)

    def load_in_thread():
        status = load_model_safe()
        print("Model status:", status)

    threading.Thread(target=load_in_thread, daemon=True).start()

    api = Api()
    window = webview.create_window(
        "Chatterbox TTS",
        str(HTML_FILE),
        js_api=api,
        width=680,
        height=780,
        min_size=(500, 600),
        background_color="#1a1a2e",
    )
    global _main_window
    _main_window = window
    webview.start(debug=False, http_server=True)


if __name__ == "__main__":
    main()