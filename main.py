"""
Main CLI Application: Dialogue Frame Finder
Extracts the exact video frame image, timestamp, and text where a target dialogue appears in a video URL.
Includes progress callback support for real-time Web UI streaming.
"""

import os
import sys
import cv2
import argparse
from src.downloader import MediaDownloader
from src.audio_aligner import AudioAligner
from src.ocr_matcher import VisualOCRMatcher

def extract_and_save_frame(video_path, frame_number, output_image_path="output_frame.png"):
    """
    Seeks directly to frame_number in the video file and saves the actual video screenshot image.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret or frame is None or frame.shape[0] == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 2))
        ret, frame = cap.read()
        
    cap.release()
    
    if ret and frame is not None:
        cv2.imwrite(output_image_path, frame)
        print(f"[+] Video screenshot frame #{frame_number} successfully saved to: '{output_image_path}'")
        return True
    else:
        print(f"[!] Warning: Could not read frame #{frame_number} from {video_path}")
        return False

def find_dialogue_frame(video_url, target_text, output_img="output_frame.png", transcript_txt="transcript.txt", transcript_json="transcript.json", progress_callback=None, skip_ocr=False, model_size="small"):
    def update_progress(percent, msg):
        if progress_callback:
            progress_callback(percent, msg)
        print(f"[{percent}%] {msg}")

    update_progress(5, f"Initializing pipeline for URL: {video_url}")

    # Step 1: Media Downloader & Metadata Ingestion
    update_progress(10, "Fetching video stream metadata via yt-dlp...")
    downloader = MediaDownloader()
    meta = downloader.download_media(video_url, progress_callback=progress_callback)
    
    fps = meta["fps"]
    video_path = meta["video_path"]
    total_frames = meta["total_frames"]

    # Early exit if download failed entirely
    if fps <= 0 or total_frames <= 0:
        err = (
            f"Video could not be downloaded or read (FPS={fps:.2f}, frames={total_frames}). "
            f"If this is an ok.ru/restricted URL, try downloading the video manually "
            f"and providing the local file path instead."
        )
        update_progress(100, f"[DOWNLOAD ERROR] {err}")
        return {"found": False, "download_error": True, "error_message": err,
                "timestamp": None, "frame": None, "text": None}

    update_progress(25, f"Stream ingested: {meta['resolution']}, {fps:.2f} FPS, {total_frames} total frames")
    
    # Step 2: Primary Audio Alignment (Whisper STT) & Full Transcript Generation
    update_progress(35, f"Extracting audio waveform and initializing Whisper STT model ('{model_size}')...")
    audio_aligner = AudioAligner(model_size)
    
    update_progress(50, f"Transcribing audio track and searching for spoken dialogue '{target_text}'...")
    audio_res = audio_aligner.find_spoken_dialogue(
        video_path,
        target_text,
        fps=fps,
        transcript_txt_path=transcript_txt,
        transcript_json_path=transcript_json,
        progress_callback=progress_callback
    )

    final_result = None

    if audio_res["found"]:
        anchor_frame = audio_res["frame"]
        update_progress(75, f"Spoken match found in audio at frame #{anchor_frame} ({audio_res['timestamp']})")

        if skip_ocr:
            update_progress(90, "OCR skipped (no-subtitles mode). Using audio anchor frame directly.")
            final_result = {
                "timestamp": audio_res["timestamp"],
                "frame": audio_res["frame"],
                "text": audio_res["text"]
            }
        else:
            update_progress(80, "Inspecting candidate 2-second frame window via EasyOCR...")
            ocr_matcher = VisualOCRMatcher()
            start_win = max(0, anchor_frame - int(fps * 1))
            end_win = min(meta["total_frames"] - 1, anchor_frame + int(fps * 1))
            ocr_res = ocr_matcher.inspect_frame_window(video_path, target_text, start_win, end_win, fps=fps)

            if ocr_res["found"]:
                update_progress(90, f"Visual on-screen subtitle confirmed at frame #{ocr_res['frame']}")
                final_result = {
                    "timestamp": ocr_res["timestamp"],
                    "frame": ocr_res["frame"],
                    "text": ocr_res["text"]
                }
            else:
                update_progress(90, f"No visual subtitle overlay; confirming spoken audio entry frame #{anchor_frame}")
                final_result = {
                    "timestamp": audio_res["timestamp"],
                    "frame": audio_res["frame"],
                    "text": audio_res["text"]
                }
    else:
        update_progress(60, "Spoken dialogue not found in audio. Commencing coarse-to-fine Visual OCR search...")
        ocr_matcher = VisualOCRMatcher()
        ocr_res = ocr_matcher.inspect_frame_window(video_path, target_text, 0, meta["total_frames"] - 1, fps=fps)
        if ocr_res["found"]:
            update_progress(90, f"Visual on-screen dialogue match located at frame #{ocr_res['frame']}")
            final_result = {
                "timestamp": ocr_res["timestamp"],
                "frame": ocr_res["frame"],
                "text": ocr_res["text"]
            }

    if final_result:
        # Calculate 6-second slice surrounding target timestamp (3.0s before, 3.0s after)
        target_frame = final_result["frame"]
        center_sec = target_frame / fps if fps > 0 else 0
        start_sec = max(0, center_sec - 3.0)
        end_sec = min(meta["duration_sec"], center_sec + 3.0)
        if (end_sec - start_sec) < 6.0 and meta["duration_sec"] >= 6.0:
            if start_sec == 0:
                end_sec = min(6.0, meta["duration_sec"])
            elif end_sec == meta["duration_sec"]:
                start_sec = max(0, meta["duration_sec"] - 6.0)

        update_progress(92, f"Fetching maximum quality 6-second video clip ({start_sec:.1f}s - {end_sec:.1f}s)...")
        clip_path = downloader.download_high_quality_clip(video_url, start_sec, end_sec, "output_clip.mp4")

        update_progress(97, f"Extracting high-resolution screenshot frame #{final_result['frame']}...")
        # Calculate offset timestamp in HQ clip: relative_sec = center_sec - start_sec
        relative_sec = max(0.0, center_sec - start_sec)
        # Extract screenshot frame directly from high-quality downloaded clip if valid
        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1024:
            cap_clip = cv2.VideoCapture(clip_path)
            clip_fps = cap_clip.get(cv2.CAP_PROP_FPS) or fps
            cap_clip.release()
            
            hq_frame_index = int(round(relative_sec * clip_fps))
            extract_and_save_frame(clip_path, hq_frame_index, output_img)
        else:
            extract_and_save_frame(video_path, final_result["frame"], output_img)

        update_progress(100, "Processing complete! Outputs ready.")

        print("\n" + "=" * 50)
        print("                  FINAL OUTPUT                  ")
        print("=" * 50)
        print(f"Timestamp   : {final_result['timestamp']}")
        print(f"Frame       : {final_result['frame']}")
        print(f"Text        : \"{final_result['text']}\"")
        print(f"HQ Image    : Saved screenshot to '{output_img}'")
        print(f"Video Clip  : Saved max quality 6s clip to '{clip_path}'")
        print(f"Transcript  : Saved to '{transcript_txt}' & '{transcript_json}'")
        print("=" * 50 + "\n")
        final_result["found"] = True
        final_result["clip_path"] = clip_path
        final_result["total_frames"] = total_frames
        final_result["fps"] = fps
        return final_result
    else:
        candidates = audio_res.get("candidates", [])
        hint = audio_res.get("hint", "")
        update_progress(100, f"Dialogue not found in video.")
        print(f"\n[!] Dialogue \"{target_text}\" could not be located.")
        if candidates:
            print(f"[*] Closest matches found:")
            for c in candidates:
                print(f"    [{c['score']:.0f}%] \"{c['text']}\" at {c['timestamp']} (frame {c['frame']})")
        print(f"[+] Full transcript saved to '{transcript_txt}'")
        if hint:
            print(f"[*] Hint: {hint}")
        return {
            "found": False,
            "timestamp": None,
            "frame": None,
            "text": None,
            "candidates": candidates,
            "hint": hint,
            "total_frames": total_frames,
            "fps": fps,
        }


def main():
    parser = argparse.ArgumentParser(description="Find exact frame where dialogue appears in a video URL.")
    parser.add_argument("--url", type=str, default="https://ok.ru/video/248244667877", help="Video URL")
    parser.add_argument("--text", type=str, default="My mind rebels at stagnation", help="Target dialogue text")
    parser.add_argument("--output", type=str, default="output_frame.png", help="Output frame image path")
    parser.add_argument("--transcript", type=str, default="transcript.txt", help="Output transcript file path")
    parser.add_argument("--model", type=str, default="small", choices=["tiny", "base", "small", "medium", "large"], help="Whisper STT model size")
    
    args = parser.parse_args()
    find_dialogue_frame(args.url, args.text, args.output, transcript_txt=args.transcript, model_size=args.model)


if __name__ == "__main__":
    main()
