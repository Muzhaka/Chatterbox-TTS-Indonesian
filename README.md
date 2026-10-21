# Chatterbox TTS Indonesia - Generate & Clone Bahasa Indonesia

> **v1.0.0** — Full AI-generated desktop app untuk TTS & voice cloning Bahasa Indonesia, menggunakan model [grandhigh/Chatterbox-TTS-Indonesian](https://huggingface.co/grandhigh/Chatterbox-TTS-Indonesian).

## 📌 Deskripsi

Aplikasi desktop PyWebView ini memungkinkan:

- **Generate Suara**: Text-to-Speech Bahasa Indonesia dengan model *fine-tuned* Chatterbox.
- **Clone Suara**: Voice cloning zero-shot dengan file audio referensi.
- **Voice Profile**: Kelola profil suara pribadi (folder-based, file voice.wav per profil).
- **Pilihan Perangkat Keras**: Dukung NVIDIA CUDA (GPU) dan CPU.
- **Auto-Batching Teks Panjang**: Teks panjang dipotong otomatis per ~600 token berdasarkan tanda baca/kata, digenerate berurutan, lalu digabung rapi tanpa suara meracau/berulang.
- **File Sumber Asli**: Semua model dan checkpoint disediakan oleh [grandhigh/Chatterbox-TTS-Indonesian](https://huggingface.co/grandhigh/Chatterbox-TTS-Indonesian) — file besar **tidak termasuk** di repo ini, tetapi diunduh otomatis saat pertama kali diperlukan.

> **Catatan**: Project ini **dibuat sepenuhnya oleh AI** melalui *vibe coding*. Semua komponen — struktur proyek, backend Python (PyWebView), frontend HTML/JS, dan skrip peluncur — disusun dan diselesaikan oleh AI agent (Claude via Hermes Agent).

## 📂 Struktur Project

```
chatterbox/
├── ChatterboxTTS.bat           # Peluncur utama (jalan dengan pythonw.exe — tak ada CMD)
├── setup.py                    # Auto-download model weights dari Hugging Face
├── chatterbox_gui.py           # Backend Python (PyWebView API)
├── gui.html                    # Frontend GUI (HTML/CSS/JS)
├── voice_profiles.json         # Profil suara (otomatis dibuat)
├── .gitignore                  # Mengabaikan venv, model, audio output
├── profile_model/              # [auto] Suara per profil → profile_model/<pid>/voice.wav
├── base_clone/                 # [auto] File sumber untuk cloning
├── tts_outputs/                # [auto] Hasil generate suara
├── output_clone/               # [auto] Hasil cloning suara
├── logs/                       # [auto] Log aplikasi
└── venv/                       # [tidak termasuk di repo] Virtual environment
```

## 🚀 Cara Pakai

1. **Clone repo ini:**
   ```bash
   git clone https://github.com/Muzhaka/chatterbox-tts-indonesia.git
   cd chatterbox-tts-indonesia
   ```

2. **Buat & aktifkan virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   source venv/bin/activate    # Linux/macOS
   ```

3. **Install dependensi:**
   ```bash
   pip install chatterbox-tts soundfile safetensors huggingface_hub pywebview numpy
   pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

4. **Unduh model otomatis (hanya pertama kali):**
   ```bash
   python setup.py
   ```
   > Atau cukup jalankan `ChatterboxTTS.bat` — model akan diunduh otomatis pada permintaan pertama.

5. **Buka aplikasi:**
   Double-click `ChatterboxTTS.bat`.

## ⚙️ Fitur

| Fitur | Deskripsi |
|-------|-----------|
| Generate Suara | Masukkan teks, pilih profil suara, generate audio |
| Clone Suara | Pilih file audio, masukkan teks, clone suara |
| Voice Profile | Kelola profil suara pribadi dengan file .wav |
| VGA / CPU Mode | Pilih penggunaan GPU (CUDA) atau CPU |
| Native File Dialog | Memilih file audio melalui sistem file dialog alami |
| Instant Shutdown | Tombol Tutup Aplikasi menutup semua komponen |

## 📜 Lisensi & Atribusi

- **Lisensi repositori ini**: MIT License
- **Model AI**: [grandhigh/Chatterbox-TTS-Indonesian](https://huggingface.co/grandhigh/Chatterbox-TTS-Indonesian), Apache 2.0 License
- **Chatterbox TTS**: Oleh [Pritam Inc](https://github.com/Tiger-Ananthan/Chatterbox)
- File model besar (`t3_cfg.safetensors`, ~5.3 GB) adalah karya dari [grandhigh](https://huggingface.co/grandhigh) dan **tidak dimodifikasi** — diunduh secara otomatis dari Hugging Face untuk menjaga agar repositori tetap ringan.

---

### ⚠️ Persyaratan Perangkat Keras

- **Rekomendasi**: GPU NVIDIA dengan CUDA 11.8 (memaksimalkan performa)
- **Minimum**: CPU (akan lebih lambat, tapi masih berfungsi)

---

**© 2026 — Dibuat penuh oleh AI (vibe coding) dengan ❤️ untuk komunitas TTS Bahasa Indonesia.**
**Model bertanggung jawab: [grandhigh/Chatterbox-TTS-Indonesian](https://huggingface.co/grandhigh/Chatterbox-TTS-Indonesian)**