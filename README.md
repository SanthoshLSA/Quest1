# Dialogue Frame Finder

Give it a video URL and a line of dialogue. It finds the exact frame in the video where that line appears, saves a screenshot, and cuts a short clip around it.

Works with YouTube, ok.ru, and most other major video platforms.

[![GitHub Repo](https://img.shields.io/badge/GitHub-SanthoshLSA%2FQuest1-blue)](https://github.com/SanthoshLSA/Quest1)

---

## How It Works (Quick Version)

1. Downloads the video in low quality (fast, ~30 seconds for a 1-hour video)
2. Runs the audio through Whisper to find *when* the dialogue is spoken
3. Looks at frames around that moment and uses OCR to confirm the exact frame
4. Downloads just that 6-second window in full HD
5. Shows you the frame number, timestamp, screenshot, and a playable clip

Full technical details are in [APPROACH.md](APPROACH.md).

---

## Requirements

- Python 3.9 or higher
- Git
- FFmpeg (installed automatically via `imageio-ffmpeg` — no manual setup needed)

> You do **not** need to install FFmpeg manually. The Python package handles it.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/SanthoshLSA/Quest1.git
cd Quest1
```

### 2. (Recommended) Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything: Whisper, EasyOCR, yt-dlp, FastAPI, OpenCV, etc.

> First run will also download EasyOCR language models (~100MB) and Whisper's `tiny` model (~75MB) automatically. You only need internet for the first run.

---

## Run via Command Line

```bash
python main.py --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation"
```

**Arguments:**
- `--url` — any publicly accessible video URL (YouTube, ok.ru, direct MP4, etc.)
- `--text` — the dialogue line you're looking for
- `--output` — (optional) filename to save the frame image, default is `output_frame.png`

**Example output:**
```
Timestamp  : 00:05:25.160
Frame      : 7797
Text Found : "My mind rebels at stagnation."
Clip       : output_clip.mp4
Screenshot : output_frame.png
```

---

## Run via Web Dashboard

```bash
python app.py
```

Then open your browser at: **http://localhost:8000**

Paste a video URL and your dialogue text, hit **Find Frame**, and watch it process in real time. When done it shows you the frame number, FPS, total frames, a screenshot, and a playable video clip.

---

## Project Structure

```
Quest1/
├── README.md              # This file
├── APPROACH.md            # Design decisions and how everything works
├── prompts.txt            # Development prompt log
├── requirements.txt       # All Python dependencies
├── main.py                # Command-line entrypoint
├── app.py                 # Web server (FastAPI + SSE streaming)
├── test_env.py            # Checks your environment is set up correctly
└── src/
    ├── downloader.py      # Downloads video/audio and extracts metadata
    ├── audio_aligner.py   # Whisper transcription and timestamp search
    └── ocr_matcher.py     # Frame OCR and fuzzy text matching
```

---

## Dependencies

| Package | What it does |
|---|---|
| `yt-dlp` | Downloads video/audio from 1000+ platforms |
| `openai-whisper` | Speech-to-text — finds when dialogue is spoken |
| `easyocr` | Reads text from video frames |
| `opencv-python` | Frame extraction and image processing |
| `rapidfuzz` | Fuzzy string matching (handles OCR typos) |
| `imageio-ffmpeg` | Bundles FFmpeg — no manual install needed |
| `fastapi` + `uvicorn` | Web dashboard backend |
| `pillow` | Image handling |
| `numpy` | Audio waveform processing |

---

## Troubleshooting

**Whisper model not downloading?**  
Make sure you have internet on first run. Models are cached to `~/.cache/whisper/` after that.

**EasyOCR slow on first run?**  
Normal — it downloads ~100MB of language models once. After that it's fast.

**Video not downloading (ok.ru / SSL errors)?**  
Update yt-dlp: `pip install -U yt-dlp`

**Output clip not playing in browser?**  
The clip is encoded as H.264 + AAC which plays in all modern browsers. If it doesn't play, try opening `output_clip.mp4` directly in VLC.
