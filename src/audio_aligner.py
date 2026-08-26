"""
audio_aligner.py
Transcribes video/audio and finds the exact timestamp where spoken dialogue appears.
Uses faster-whisper (CTranslate2 engine) for 4-8x faster transcription vs openai-whisper.
Falls back to openai-whisper if faster-whisper is not installed.
Supports early exit — stops as soon as the target dialogue is found.
"""

import os
import json
import subprocess
import numpy as np
from rapidfuzz import fuzz

MATCH_THRESHOLD = 60.0  # minimum similarity score (0-100) to accept a match


def load_audio_numpy(file_path, sr=16000):
    """Loads audio from any video/audio file into a 16kHz mono float32 numpy array."""
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-i", file_path,
        "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, _ = process.communicate()
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio


def _try_load_faster_whisper(model_size):
    """Attempts to load a faster-whisper model. Returns model or None."""
    try:
        from faster_whisper import WhisperModel
        print(f"[*] Loading faster-whisper model ('{model_size}')...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"[+] faster-whisper loaded (int8 quantized, CPU).")
        return model, "faster_whisper"
    except ImportError:
        return None, None


def _try_load_openai_whisper(model_size):
    """Loads the standard openai-whisper model."""
    import whisper
    print(f"[*] Loading openai-whisper model ('{model_size}')...")
    model = whisper.load_model(model_size)
    print(f"[+] openai-whisper loaded.")
    return model, "openai_whisper"


class AudioAligner:
    def __init__(self, model_size="tiny"):
        model, backend = _try_load_faster_whisper(model_size)
        if model is None:
            print("[!] faster-whisper not found, falling back to openai-whisper...")
            model, backend = _try_load_openai_whisper(model_size)
        self.model = model
        self.backend = backend
        self.model_size = model_size

    def find_spoken_dialogue(
        self, media_path, target_text, fps=23.98,
        transcript_txt_path="transcript.txt",
        transcript_json_path="transcript.json",
        progress_callback=None
    ):
        """
        Transcribes the audio and searches for target_text.
        Stops as soon as a match is found (early exit).
        Returns the timestamp, frame number, matched text, and top near-miss candidates
        even when no match is found.
        """
        def notify(pct, msg):
            print(f"[{pct}%] {msg}")
            if progress_callback:
                progress_callback(pct, msg)

        print(f"[*] Extracting audio from: {media_path}")
        audio_np = load_audio_numpy(media_path, sr=16000)
        duration_sec = len(audio_np) / 16000.0
        print(f"[*] Audio duration: {duration_sec:.1f}s — transcribing with {self.backend}...")

        target_clean = target_text.lower().strip()
        all_segments = []
        best_match = None
        best_score = 0.0

        # ── faster-whisper path (streaming segments, early exit possible) ──
        if self.backend == "faster_whisper":
            segments_iter, info = self.model.transcribe(
                audio_np,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,          # skip silent sections entirely
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=False,
            )
            for seg in segments_iter:
                seg_data = {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip()
                }
                all_segments.append(seg_data)

                # Live progress
                pct_done = min(seg.end / duration_sec, 1.0)
                overall = int(35 + pct_done * 35)
                notify(overall, f"Transcribing: {int(pct_done * 100)}% — {seg.text.strip()[:60]}")

                # Early exit check
                score = fuzz.partial_ratio(target_clean, seg.text.lower().strip())
                if score > best_score:
                    best_score = score
                    best_match = seg_data
                if score >= MATCH_THRESHOLD:
                    print(f"[+] Early exit — match found at {seg.start:.1f}s (score {score:.0f}%)")
                    # Drain remaining segments silently for the transcript
                    for remaining in segments_iter:
                        all_segments.append({
                            "start": remaining.start,
                            "end": remaining.end,
                            "text": remaining.text.strip()
                        })
                    break

        # ── openai-whisper path (full transcription, then search) ──
        else:
            import sys
            import re

            class TqdmCapture:
                def __init__(self, cb, dur):
                    self.cb = cb
                    self.dur = dur
                def write(self, text):
                    text = text.strip()
                    if not text or not self.cb:
                        return
                    m = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)", text)
                    if m:
                        pct = int(m.group(1))
                        overall = int(35 + (pct / 100.0) * 35)
                        self.cb(overall, f"Transcribing audio: {pct}%")
                def flush(self):
                    pass

            orig_err = sys.stderr
            if progress_callback:
                sys.stderr = TqdmCapture(progress_callback, duration_sec)
            try:
                result = self.model.transcribe(
                    audio_np,
                    verbose=False, fp16=False,
                    beam_size=1, best_of=1,
                    condition_on_previous_text=False,
                    temperature=0.0
                )
            finally:
                sys.stderr = orig_err

            all_segments = result.get("segments", [])
            for seg in all_segments:
                score = fuzz.partial_ratio(target_clean, seg["text"].lower().strip())
                if score > best_score:
                    best_score = score
                    best_match = seg

        notify(70, f"Transcription done — {len(all_segments)} segments processed")

        # ── Save full transcript ──
        self._save_transcript(all_segments, fps, transcript_txt_path, transcript_json_path)

        # ── Build near-miss candidates (top 3 closest lines) for "not found" UX ──
        scored = sorted(
            [{"seg": s, "score": fuzz.partial_ratio(target_clean, s["text"].lower().strip())}
             for s in all_segments],
            key=lambda x: x["score"], reverse=True
        )
        top_candidates = [
            {
                "text": c["seg"]["text"].strip(),
                "score": c["score"],
                "timestamp": self._fmt_ts(c["seg"]["start"]),
                "frame": int(round(c["seg"]["start"] * fps))
            }
            for c in scored[:3]
        ]

        full_text = " ".join(s["text"] for s in all_segments).strip()

        if best_match and best_score >= MATCH_THRESHOLD:
            start_time = best_match["start"]
            ts = self._fmt_ts(start_time)
            frame_num = int(round(start_time * fps))
            print(f"\n[+] Match found: \"{best_match['text'].strip()}\" at {ts} (score {best_score:.0f}%)")
            return {
                "found": True,
                "start_time": start_time,
                "timestamp": ts,
                "frame": frame_num,
                "text": best_match["text"].strip(),
                "score": best_score,
                "full_transcript_text": full_text,
                "candidates": top_candidates
            }
        else:
            print(f"\n[!] Dialogue not found in audio (best score: {best_score:.0f}%)")
            print(f"    Closest match: \"{top_candidates[0]['text'] if top_candidates else 'none'}\"")
            return {
                "found": False,
                "score": best_score,
                "full_transcript_text": full_text,
                "candidates": top_candidates,
                "hint": (
                    f"No segment scored above {MATCH_THRESHOLD:.0f}%. "
                    f"Best was {best_score:.0f}% — '{top_candidates[0]['text'] if top_candidates else ''}'. "
                    f"Check transcript.txt for the full transcript and try adjusting the search text."
                )
            }

    def _fmt_ts(self, secs):
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def _save_transcript(self, segments, fps, txt_path, json_path):
        lines = []
        json_segs = []
        for seg in segments:
            ts = self._fmt_ts(seg["start"])
            lines.append(f"[{ts}] {seg['text'].strip()}")
            json_segs.append({
                "start_seconds": seg["start"],
                "end_seconds": seg["end"],
                "timestamp": ts,
                "frame": int(round(seg["start"] * fps)),
                "text": seg["text"].strip()
            })
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== FULL VIDEO TRANSCRIPT ===\n\n" + "\n".join(lines))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"segment_count": len(segments), "segments": json_segs}, f, indent=2)
        print(f"[+] Transcript saved: {txt_path}, {json_path}")


if __name__ == "__main__":
    media_file = "downloads/input_video.mp4"
    if os.path.exists(media_file):
        aligner = AudioAligner("tiny")
        aligner.find_spoken_dialogue(media_file, "My mind rebels at stagnation")
    else:
        print("[!] Video file not found. Run downloader first.")
