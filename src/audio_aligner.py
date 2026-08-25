"""
Module: audio_aligner.py
Transcribes video/audio using Whisper and locates exact spoken dialogue timestamp and frame number using fuzzy matching.
Pure Python audio loading via scipy/wave to bypass system ffmpeg requirement.
"""

import os
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

    def find_spoken_dialogue(self, media_path, target_text, fps=23.98):
        """
        Transcribes media file, searches segments for target_text using RapidFuzz,
        and returns exact starting timestamp (sec), timestamp string (HH:MM:SS.sss), and frame number.
        """
        print(f"[*] Extracting audio waveform from: {media_path}...")
        audio_np = load_audio_numpy(media_path, sr=16000)
        
        print(f"[*] Transcribing audio ({len(audio_np)/16000:.1f}s) with Whisper...")
        result = self.model.transcribe(audio_np, verbose=False, fp16=False)
        
        segments = result.get("segments", [])
        print(f"[+] Transcription finished. Analyzed {len(segments)} audio segments.")

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
                "score": best_score
            }
        else:
            print(f"[!] Target dialogue not found in audio track (Best score: {best_score:.1f}%).")
            return {"found": False, "score": best_score}


if __name__ == "__main__":
    media_file = "downloads/input_video.mp4"
    if os.path.exists(media_file):
        aligner = AudioAligner("tiny")
        aligner.find_spoken_dialogue(media_file, "My mind rebels at stagnation")
    else:
        print("[!] Video file not found. Run downloader first.")
