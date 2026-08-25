# Dialogue Frame Finder

An automated, multi-modal Python application that finds the **exact video frame**, timestamp (`HH:MM:SS.sss`), extracted text, and saves the frame image where a dialogue first appears in a media URL.

[![GitHub Repo](https://img.shields.io/badge/GitHub-SanthoshLSA%2FQuest1-blue)](https://github.com/SanthoshLSA/Quest1)

---

## Output Example

When run on the evaluation URL `https://ok.ru/video/248244667877` for the dialogue `"My mind rebels at stagnation"`:

```text
==================================================
                  FINAL OUTPUT                  
==================================================
Timestamp : 00:05:25.160
Frame     : 7797
Text      : "My mind reveals its stagnation."
Image     : Saved to 'output_frame.png'
==================================================
```

The corresponding video frame is extracted and saved to `output_frame.png`.

---

## Repository Structure

```text
.
├── README.md           # Instructions to set up and run the solution
├── APPROACH.md         # Detailed design choices, algorithms, trade-offs, and edge case handling
├── prompts.txt          # Complete structured human prompt log across all development steps
├── requirements.txt    # Project dependencies (opencv-python, yt-dlp, easyocr, rapidfuzz, openai-whisper, pillow)
├── main.py             # Main CLI execution entrypoint
├── test_env.py         # Environment sanity check script
└── src/
    ├── downloader.py   # Ingests video metadata and streams using yt-dlp
    ├── audio_aligner.py # Primary Whisper speech-to-text alignment engine
    └── ocr_matcher.py  # EasyOCR text detection & RapidFuzz matching engine
```

---

## Installation & Setup Instructions

### Prerequisites
- Python 3.9+
- Git

### 1. Clone Repository
```bash
git clone https://github.com/SanthoshLSA/Quest1.git
cd Quest1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

Execute `main.py` passing `--url` and `--text`:

```bash
python main.py --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation"
```

### Options:
- `--url`: Publicly accessible video URL (e.g. ok.ru, YouTube, or direct MP4/stream link).
- `--text`: Target dialogue string to search for.
- `--output`: Path to save the extracted frame image (default: `output_frame.png`).

---

## Architecture Overview

Our solution utilizes an **Audio-Anchored Dual-Modal Engine**:
1. **Primary Fast Audio Anchor (Whisper STT)**: Scans the audio track in ~3 minutes to locate the candidate dialogue timestamp window $t_{\text{dialogue}}$, avoiding wasteful scanning of 78,000+ video frames.
2. **Fine Visual Frame Pinpointer (EasyOCR + CLAHE)**: Inspects the candidate frame window around $t_{\text{dialogue}}$ using OpenCV contrast enhancement and EasyOCR to detect on-screen subtitles/text overlays.
3. **Coarse-to-Fine Fallback**: If no audio speech match exists (e.g., silent video or title card), the system automatically performs full video coarse-to-fine visual OCR search.

For deep technical details, mathematical formulas, trade-off matrices, and candidate evaluation analysis, see **[APPROACH.md](APPROACH.md)**.
