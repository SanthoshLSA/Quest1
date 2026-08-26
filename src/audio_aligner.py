"""
Module: audio_aligner.py
Transcribes video/audio using Whisper and locates exact spoken dialogue timestamp and frame number using fuzzy matching.
Pure Python audio loading via scipy/wave to bypass system ffmpeg requirement.
Saves full timestamped transcript to file.
"""

import os
import json
import wave
import subprocess
import whisper
import numpy as np
from rapidfuzz import fuzz

def load_audio_numpy(file_path, sr=16000):
    """
    Loads audio file directly into 16kHz mono float32 numpy array without system ffmpeg.
    """
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    cmd = [
        ffmpeg_exe,
        "-i", file_path,
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(sr),
        "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, _ = process.communicate()
    
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio

class AudioAligner:
    def __init__(self, model_size="tiny"):
        print(f"[*] Initializing Whisper model ('{model_size}')...")
        self.model = whisper.load_model(model_size)

    def find_spoken_dialogue(self, media_path, target_text, fps=23.98, transcript_txt_path="transcript.txt", transcript_json_path="transcript.json", progress_callback=None):
        """
        Transcribes media file, searches segments for target_text using RapidFuzz,
        saves the ENTIRE transcript to transcript.txt and transcript.json,
        and returns exact starting timestamp (sec), timestamp string (HH:MM:SS.sss), and frame number.
        """
        print(f"[*] Extracting audio waveform from: {media_path}...")
        audio_np = load_audio_numpy(media_path, sr=16000)
        
        duration_sec = len(audio_np) / 16000.0
        print(f"[*] Transcribing audio ({duration_sec:.1f}s) with Whisper...")

        # Stream live tqdm stats during Whisper transcription
        class TqdmProgressStream:
            def __init__(self, callback, duration):
                self.callback = callback
                self.duration = duration
                self.last_pct = -1

            def write(self, text):
                text = text.strip()
                if not text or not self.callback:
                    return
                # Match tqdm pattern: 39%|███▌     | 22016/55883 [00:21<00:38, 882.20frames/s]
                import re
                match = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+frames/s)\]", text)
                if match:
                    pct, cur_f, tot_f, elapsed, remaining, speed = match.groups()
                    pct_val = int(pct)
                    overall_pct = int(35 + (pct_val / 100.0) * 35) # Map STT 0-100% to overall 35%-70%
                    msg = f"Transcribing audio: {pct}% | {cur_f}/{tot_f} frames [{elapsed}<{remaining}, {speed}]"
                    if overall_pct != self.last_pct:
                        self.last_pct = overall_pct
                        self.callback(overall_pct, msg)

            def flush(self):
                pass

        import sys
        stdout_orig = sys.stderr
        if progress_callback:
            sys.stderr = TqdmProgressStream(progress_callback, duration_sec)

        try:
            result = self.model.transcribe(
                audio_np,
                verbose=False,
                fp16=False,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                temperature=0.0
            )
        finally:
            if progress_callback:
                sys.stderr = stdout_orig
        
        segments = result.get("segments", [])
        full_text = result.get("text", "").strip()
        print(f"[+] Transcription finished. Analyzed {len(segments)} audio segments.")

        # Save Full Transcript Files
        formatted_lines = []
        json_segments = []
        
        for seg in segments:
            start_t = seg["start"]
            end_t = seg["end"]
            hrs = int(start_t // 3600)
            mins = int((start_t % 3600) // 60)
            secs = start_t % 60
            ts_str = f"[{hrs:02d}:{mins:02d}:{secs:06.3f}]"
            
            line = f"{ts_str} {seg['text'].strip()}"
            formatted_lines.append(line)
            
            json_segments.append({
                "start_seconds": start_t,
                "end_seconds": end_t,
                "timestamp": ts_str,
                "frame": int(round(start_t * fps)),
                "text": seg["text"].strip()
            })

        # Write transcript.txt
        with open(transcript_txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== FULL VIDEO TRANSCRIPT ===\n\n")
            f.write("\n".join(formatted_lines))
        print(f"[+] Full video transcript text saved to: '{transcript_txt_path}'")

        # Write transcript.json
        with open(transcript_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "full_text": full_text,
                "segment_count": len(segments),
                "segments": json_segments
            }, f, indent=2)
        print(f"[+] Full video transcript JSON saved to: '{transcript_json_path}'")

        # Find best segment matching target dialogue
        best_segment = None
        best_score = 0.0
        target_clean = target_text.lower().strip()

        for seg in segments:
            seg_text = seg["text"].lower().strip()
            score = fuzz.partial_ratio(target_clean, seg_text)
            if score > best_score:
                best_score = score
                best_segment = seg

        if best_segment and best_score >= 60.0:
            start_time = best_segment["start"]
            frame_num = int(round(start_time * fps))
            
            hrs = int(start_time // 3600)
            mins = int((start_time % 3600) // 60)
            secs = start_time % 60
            timestamp_str = f"{hrs:02d}:{mins:02d}:{secs:06.3f}"
            extracted_text = best_segment["text"].strip()

            print(f"\n[+] Spoken Dialogue Found in Audio!")
            print(f"    - Segment Text  : \"{extracted_text}\"")
            print(f"    - Match Score   : {best_score:.1f}%")
            print(f"    - Start Time    : {start_time:.3f} sec")
            print(f"    - Timestamp     : {timestamp_str}")
            print(f"    - Start Frame   : {frame_num}\n")

            return {
                "found": True,
                "start_time": start_time,
                "timestamp": timestamp_str,
                "frame": frame_num,
                "text": extracted_text,
                "score": best_score,
                "full_transcript_text": full_text
            }
        else:
            print(f"[!] Target dialogue not found in audio track (Best score: {best_score:.1f}%).")
            return {"found": False, "score": best_score, "full_transcript_text": full_text}


if __name__ == "__main__":
    media_file = "downloads/input_video.mp4"
    if os.path.exists(media_file):
        aligner = AudioAligner("tiny")
        aligner.find_spoken_dialogue(media_file, "My mind rebels at stagnation")
    else:
        print("[!] Video file not found. Run downloader first.")
