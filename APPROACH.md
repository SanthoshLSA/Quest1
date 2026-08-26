# Approach & Design: Dialogue Frame Finder

**Repo**: [SanthoshLSA/Quest1](https://github.com/SanthoshLSA/Quest1)  
**Author**: Santhosh  
**What it does**: You give it a video URL and a line of dialogue. It finds the exact frame in the video where that line appears and gives you a screenshot and a short clip.

---

## The Problem

Finding a specific line of dialogue in a long video is harder than it sounds. A one-hour video at 25fps has 90,000 frames. Reading text from every single frame with OCR would take several hours on a normal laptop. We needed something smarter.

There were also platform challenges — sites like ok.ru protect their video streams with authentication tokens and custom headers. Standard tools like OpenCV can't open those URLs directly.

---

## How We Approached It

### Step 1: Listen First, Look Second

The key insight was this: audio is much cheaper to process than video frames.

Instead of looking at every frame visually, we first run the audio track through Whisper (a speech recognition model). Whisper can transcribe an entire hour of audio in about a minute. Once we know *when* the dialogue is spoken, we only need to inspect a small window of frames around that moment — maybe 3–6 seconds worth — rather than the entire video.

This cuts the visual work down by about 99%.

**Primary path**: Whisper finds the spoken dialogue timestamp → we look at frames only in that window → OCR confirms the exact frame with the subtitle text.

**Fallback path**: If Whisper finds nothing (silent video, or text appears on screen without being spoken), we fall back to a broader OCR scan of the video in chunks.

---

## Key Decisions Made

### Downloading the Video

**Why yt-dlp?**  
Most video platforms (ok.ru, YouTube, etc.) don't give you a simple direct file link. They use signed tokens, HLS manifests, and referrer checks. yt-dlp handles all of that automatically. OpenCV's built-in URL reader can't deal with this — it just gets a 403 error.

**Two-phase download strategy** (latest update):  
We used to download the full video in highest quality right away. For a 1-hour video, that's 1GB+ and takes several minutes. Now we do it in two phases:

1. **Initial download in low quality** (smallest/worst format, ~30–80MB) — just enough to run Whisper and find *where* the dialogue is. This downloads in 20–40 seconds.
2. **Final clip download in max quality** — once we know the exact 6-second window, we download *only that slice* from the remote source at full resolution using `yt-dlp`'s `download_ranges` feature. This means the clip and screenshot are always sharp and high quality, even though the full video was never fully downloaded in HD.

This approach gives you fast results AND a high-quality output.

---

### Speech Recognition (Whisper)

We use OpenAI Whisper (`tiny` model by default). It runs entirely offline — no API key, no cloud call.

**Speed optimizations added**:  
By default Whisper uses beam search (checks multiple possible word sequences). We turned that off:
- `beam_size=1` — greedy decoding only, 3–4× faster
- `best_of=1` — no sampling fallback
- `condition_on_previous_text=False` — treats each chunk independently, avoids slow context carry-over
- `temperature=0.0` — deterministic output

For finding dialogue in a movie, this speed tradeoff is totally fine. Accuracy stays high for clear speech.

---

### Reading Text from Frames (OCR)

We use EasyOCR to read text from video frames. Video frames are often blurry or compressed, so we pre-process them with CLAHE (a contrast enhancement filter) before feeding to OCR. This helps it read subtitles that would otherwise be missed.

We don't require an exact character-for-character match. We use RapidFuzz's partial ratio (Levenshtein distance) with a 65% similarity threshold. This handles common OCR errors like `l` vs `1`, `rn` vs `m`, etc.

---

### The Web UI

The web dashboard runs on FastAPI with a simple black-and-white terminal-style interface. When you submit a URL and dialogue text:

1. It streams live progress to your browser via Server-Sent Events (SSE) — you see each step as it happens.
2. When done, it shows you the matched frame number, total frames, FPS, timestamp, and displays the screenshot and 6-second video clip.
3. The clip is encoded in H.264 + AAC so it plays natively in every browser without plugins.

---

## What We Rejected and Why

| Option | Why We Rejected It |
|---|---|
| OpenCV direct URL | Can't handle platform authentication — just returns errors on ok.ru, YouTube, etc. |
| Selenium/Playwright browser | Huge overhead, slow, hard to extract frame-accurate pixel data |
| Full video brute-force OCR | Would take 3–5+ hours on a 1-hour video |
| Cloud Vision APIs (GPT-4, Gemini) | Costs money, needs internet, non-deterministic output |
| Downloading full HD video upfront | 1GB+ download takes too long — fixed with two-phase download |
| Whisper beam search (default) | 3–4× slower than greedy decoding, unnecessary for this use case |

---

## Accuracy & Matching

Frame number is calculated as:

```
frame_number = floor(timestamp_in_seconds × fps)
```

Timestamps come from Whisper segment starts, accurate to roughly ±0.5 seconds. OCR fine-tuning within the local window narrows this further to the specific frame where text appears on screen.

---

## Edge Cases Handled

- **No subtitles on screen** — falls back to the audio timestamp frame directly
- **Silent video / title cards only** — skips audio and runs visual OCR scan
- **OCR noise / typos** — fuzzy matching absorbs minor character errors
- **ok.ru SSL quirks** — retries with Referer header and certificate checks disabled
- **Broken partial downloads** — if the targeted 6s clip download fails, falls back to trimming the local low-res copy

---

## Sample Result

Running on `https://ok.ru/video/248244667877` for the query `"My mind rebels at stagnation"`:

```
Timestamp  : 00:05:25.160
Frame      : 7797
Text Found : "My mind rebels at stagnation."
Clip       : output_clip.mp4 (6-second H.264 clip, browser-playable)
Screenshot : output_frame.png
```
