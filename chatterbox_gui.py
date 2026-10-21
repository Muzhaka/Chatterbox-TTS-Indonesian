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
import torch.nn.functional as F
import soundfile as sf
import webview

from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ── Config ──────────────────────────────────────
MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
CHECKPOINT_FILE = "t3_cfg.safetensors"

# Batas aman per batch. Di bawah batas ini model tidak mengulang/mengomong racun.
# 600 token ≈ 24 detik audio — jauh di bawah 1000 (batas default library yang "~40 detik"),
# tapi sudah cukup untuk Continuity yang natural.
BATCH_MAX_TOKENS = 600
TOKENS_PER_SECOND = 25

# Fallback conservative: teks pendek tetap generate normal (1 batch).
DEFAULT_MAX_SECONDS = 300

# Character estimation per word (rough) — model token text ~1.6 char/token Bahasa Indonesia
CHARS_PER_TEXT_TOKEN = 2.4


def estimate_text_tokens(text):
    """Rough estimate: how many text tokens this string will become."""
    return max(1, int(len(text.strip()) / CHARS_PER_TEXT_TOKEN))


def split_text_for_batches(text, batch_max_tokens=BATCH_MAX_TOKENS):
    """
    Split long text into sequential batches safe for the model.

    Strategy:
      1. Split by sentence boundary (punc . ! ? ; newline) — keeps prosody natural.
      2. Group sentences up to batch_max_tokens.
      3. If a single sentence exceeds the limit, split by comma, then by word,
         never mid-word (prevents consonant repetition artifacts).

    Returns: List[str] — each <= batch_max_tokens (approx), ordered.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Quick path: short text, no batching needed.
    if estimate_text_tokens(text) <= batch_max_tokens:
        return [text]

    # 1. Split by sentence-ending punctuation.
    #    Keep the delimiter with the preceding chunk.
    import re
    sentences = re.split(r'(?<=[.!?;\n])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        # Single giant "sentence" — fall through to word-level splitting.
        sentences = [text]

    # 2. Group by sentence into batches.
    batches = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = estimate_text_tokens(sentence)

        if s_tokens <= batch_max_tokens:
            # Fits — append to current batch if room, else flush.
            if current_tokens + s_tokens <= batch_max_tokens:
                current.append(sentence)
                current_tokens += s_tokens
            else:
                if current:
                    batches.append(" ".join(current))
                current = [sentence]
                current_tokens = s_tokens
        else:
            # Sentence too long — flush current, then split the long sentence by comma.
            if current:
                batches.append(" ".join(current))
                current = []
                current_tokens = 0
            batches.extend(_split_long_sentence(sentence, batch_max_tokens))

    # Flush trailing.
    if current:
        batches.append(" ".join(current))

    return [b for b in batches if b.strip()]


def _split_long_sentence(sentence, batch_max_tokens):
    """
    Split one oversized sentence into sub-batches.
    Order: comma → semicolon → word boundary. Never mid-word.
    """
    import re

    # Try comma
    parts = re.split(r'(?<=,)\s+', sentence)
    if len(parts) <= 1:
        # No comma — split by word
        return _split_by_words(sentence, batch_max_tokens)

    batches = []
    current = []
    current_tokens = 0

    for part in parts:
        p_tokens = estimate_text_tokens(part)
        if current_tokens + p_tokens <= batch_max_tokens:
            current.append(part)
            current_tokens += p_tokens
        else:
            if current:
                batches.append(" ".join(current))
            # This single part may still be too long — split by words.
            if p_tokens > batch_max_tokens:
                batches.extend(_split_by_words(part, batch_max_tokens))
                current = []
                current_tokens = 0
            else:
                current = [part]
                current_tokens = p_tokens

    if current:
        batches.append(" ".join(current))

    return batches


def _split_by_words(text, batch_max_tokens):
    """
    Split by word boundaries — never mid-word.
    Returns sub-batches <= batch_max_tokens (approx).
    """
    words = text.split()
    batches = []
    current = []
    current_tokens = 0

    for word in words:
        w_tokens = estimate_text_tokens(word)
        if current_tokens + w_tokens <= batch_max_tokens:
            current.append(word)
            current_tokens += w_tokens
        else:
            if current:
                batches.append(" ".join(current))
            current = [word]
            current_tokens = w_tokens

    if current:
        batches.append(" ".join(current))

    return batches


def merge_wav_files(wav_paths, output_path, sample_rate):
    """
    Concatenate multiple wav files into one, inserting 150ms silence between them.
    Prevents abrupt clicks/word-merging artifacts at batch boundaries.
    """
    import numpy as np
    import soundfile as sf

    silence = np.zeros(int(sample_rate * 0.15), dtype=np.float32)
    combined = []

    for i, path in enumerate(wav_paths):
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        combined.append(wav)
        if i < len(wav_paths) - 1:
            combined.append(silence)

    merged = np.concatenate(combined)
    sf.write(output_path, merged, sample_rate)
    return merged


def seconds_to_max_tokens(seconds):
    """Convert desired output duration (seconds) to speech-token budget."""
    return max(1000, int(seconds * TOKENS_PER_SECOND))


class ExtendedChatterbox(ChatterboxTTS):
    """
    ChatterboxTTS dengan batas durasi output yang bisa diatur.

    Library asli (chatterbox/tts.py) hardcode `max_new_tokens=1000`, yang menghasilkan
    audio ~40 detik lalu terpotong. Subclass ini membuka parameter tersebut.
    """

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_seconds=None,
    ):
        if max_seconds is None:
            max_seconds = DEFAULT_MAX_SECONDS
        max_new_tokens = seconds_to_max_tokens(max_seconds)

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, (
                "Please `prepare_conditionals` first or specify `audio_prompt_path`"
            )

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < 6561]
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        result = torch.from_numpy(watermarked_wav).unsqueeze(0)
        duration_sec = round(len(watermarked_wav) / self.sr, 1)
        result.duration_sec = duration_sec
        result.hit_token_limit = duration_sec >= max_seconds - 3
        return result

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
            m = ExtendedChatterbox.from_pretrained(device=target_device)
            ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE)
            t3_state = load_file(ckpt_path, device="cpu")
            if "model" in t3_state.keys():
                t3_state = t3_state["model"][0]
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
    def _run_batched_generation(self, text, gen_kwargs, out_file, label="batch"):
        """
        Generate audio in sequential batches for long text.

        Splits text into safe chunks (~600 tokens each), generates each chunk
        in order using the SAME voice reference, saves temp wavs, concatenates
        with 150ms silence, then cleans up temp files.

        Returns dict with {ok, batches, duration_sec, file}.
        """
        import tempfile
        import shutil

        batches = split_text_for_batches(text)
        total = len(batches)
        if total == 0:
            return {"ok": False, "error": "Teks kosong setelah dipecah"}

        tmp_dir = Path(tempfile.mkdtemp(prefix="tts_batch_"))
        try:
            tmp_files = []
            for i, chunk in enumerate(batches, start=1):
                print(f"[{label}] batch {i}/{total} ({estimate_text_tokens(chunk)} tok): "
                      f"{chunk[:60]}{'...' if len(chunk) > 60 else ''}")

                kwargs = dict(gen_kwargs)
                kwargs["text"] = chunk
                kwargs["max_seconds"] = DEFAULT_MAX_SECONDS

                wav = model.generate(**kwargs)
                wav_np = wav.squeeze().cpu().numpy()
                chunk_path = tmp_dir / f"batch_{i:03d}.wav"
                sf.write(str(chunk_path), wav_np, model.sr)
                tmp_files.append(str(chunk_path))

            # Concatenate all batches into the final output file.
            merged = merge_wav_files(tmp_files, str(out_file), model.sr)
            duration_sec = round(len(merged) / model.sr, 1)

            return {
                "ok": True,
                "batches": total,
                "duration_sec": duration_sec,
                "file": str(out_file),
            }
        finally:
            # Always clean up temp batch directory.
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
                "repetition_penalty": float(p_params.get("repetition_penalty", 1.2)),
                "min_p": float(p_params.get("min_p", 0.05)),
                "top_p": float(p_params.get("top_p", 1.0)),
                "exaggeration": float(p_params.get("exaggeration", 0.5)),
                "cfg_weight": float(p_params.get("cfg_weight", 0.5)),
                "temperature": float(p_params.get("temperature", 0.8)),
            }
            if ref_path:
                gen_kwargs["audio_prompt_path"] = ref_path

            result = self._run_batched_generation(
                text.strip(), gen_kwargs, out_file, label="generate"
            )
            if not result.get("ok"):
                return result

            res = _data_url(out_file)
            if not res.get("ok"):
                return res
            return {
                "ok": True,
                "file": str(out_file),
                "filename": out_file.name,
                "size_kb": round(out_file.stat().st_size / 1024, 1),
                "duration_sec": result.get("duration_sec"),
                "batches": result.get("batches"),
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
                "audio_prompt_path": ref_path,
                "repetition_penalty": float(p_params.get("repetition_penalty", 1.2)),
                "min_p": float(p_params.get("min_p", 0.05)),
                "top_p": float(p_params.get("top_p", 1.0)),
                "exaggeration": float(p_params.get("exaggeration", 0.5)),
                "cfg_weight": float(p_params.get("cfg_weight", 0.5)),
                "temperature": float(p_params.get("temperature", 0.8)),
            }

            result = self._run_batched_generation(
                text.strip(), gen_kwargs, out_file, label="clone"
            )
            if not result.get("ok"):
                return result

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
                "duration_sec": result.get("duration_sec"),
                "batches": result.get("batches"),
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