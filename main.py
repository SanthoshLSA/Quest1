"""
Main CLI Application: Dialogue Frame Finder
Extracts the exact video frame, timestamp, and text where a target dialogue appears in a video URL.
Now outputs full video transcript to transcript.txt and transcript.json.
"""

import os
import sys
import cv2
import argparse
from PIL import Image, ImageDraw
from src.downloader import MediaDownloader
from src.audio_aligner import AudioAligner
from src.ocr_matcher import VisualOCRMatcher

def extract_and_save_frame(video_path, frame_number, timestamp_str, text_content, output_image_path="output_frame.png"):
    """
    Extracts a specific frame number from video and saves it as an image file.
    If video stream is audio-only, generates a clean frame visualization image.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None and frame.shape[0] > 0:
        cv2.imwrite(output_image_path, frame)
        print(f"[+] Extracted frame #{frame_number} saved successfully to: {output_image_path}")
        return frame
    else:
        # Generate clean frame card image for output
        img = Image.new('RGB', (960, 540), color=(20, 24, 33))
        draw = ImageDraw.Draw(img)
        
        title_text = "DIALOGUE FRAME IDENTIFIED"
        info_text = f"Timestamp : {timestamp_str}\nFrame     : #{frame_number}\nText      : \"{text_content}\""
        
        draw.text((40, 40), title_text, fill=(255, 200, 50))
        draw.text((40, 120), info_text, fill=(240, 240, 240))
        
        img.save(output_image_path)
        print(f"[+] Output frame visualization saved successfully to: {output_image_path}")
        return img

def find_dialogue_frame(video_url, target_text, output_img="output_frame.png", transcript_txt="transcript.txt", transcript_json="transcript.json"):
    print("=" * 80)
    print("          DIALOGUE FRAME FINDER - MULTI-MODAL PIPELINE          ")
    print("=" * 80)
    print(f"Target Video URL  : {video_url}")
    print(f"Target Dialogue   : \"{target_text}\"\n")

    # Step 1: Media Downloader & Metadata Ingestion
    downloader = MediaDownloader()
    meta = downloader.download_media(video_url)
    
    fps = meta["fps"]
    video_path = meta["video_path"]
    
    # Step 2: Primary Audio Alignment (Whisper STT) & Full Transcript Generation
    audio_aligner = AudioAligner("tiny")
    audio_res = audio_aligner.find_spoken_dialogue(
        video_path,
        target_text,
        fps=fps,
        transcript_txt_path=transcript_txt,
        transcript_json_path=transcript_json
    )

    final_result = None

    if audio_res["found"]:
        anchor_frame = audio_res["frame"]
        print(f"[*] Audio Anchor Found at Frame #{anchor_frame} ({audio_res['timestamp']})")
        print(f"[*] Inspecting 5-second candidate window around audio anchor for on-screen text...")
        
        ocr_matcher = VisualOCRMatcher()
        start_win = max(0, anchor_frame - int(fps * 2))
        end_win = min(meta["total_frames"] - 1, anchor_frame + int(fps * 3))
        
        ocr_res = ocr_matcher.inspect_frame_window(video_path, target_text, start_win, end_win, fps=fps)
        
        if ocr_res["found"]:
            print(f"[+] Visual Subtitle Text Confirmed!")
            final_result = {
                "timestamp": ocr_res["timestamp"],
                "frame": ocr_res["frame"],
                "text": ocr_res["text"]
            }
        else:
            print(f"[+] No on-screen text overlay found; using exact spoken audio entry frame.")
            final_result = {
                "timestamp": audio_res["timestamp"],
                "frame": audio_res["frame"],
                "text": audio_res["text"]
            }
    else:
        print(f"[!] Spoken dialogue not detected in audio. Running full coarse-to-fine Visual OCR search...")
        ocr_matcher = VisualOCRMatcher()
        ocr_res = ocr_matcher.inspect_frame_window(video_path, target_text, 0, meta["total_frames"] - 1, fps=fps)
        if ocr_res["found"]:
            final_result = {
                "timestamp": ocr_res["timestamp"],
                "frame": ocr_res["frame"],
                "text": ocr_res["text"]
            }

    if final_result:
        # Save output image frame
        extract_and_save_frame(video_path, final_result["frame"], final_result["timestamp"], final_result["text"], output_img)
        
        print("\n" + "=" * 50)
        print("                  FINAL OUTPUT                  ")
        print("=" * 50)
        print(f"Timestamp   : {final_result['timestamp']}")
        print(f"Frame       : {final_result['frame']}")
        print(f"Text        : \"{final_result['text']}\"")
        print(f"Image       : Saved to '{output_img}'")
        print(f"Transcript  : Saved to '{transcript_txt}' & '{transcript_json}'")
        print("=" * 50 + "\n")
        return final_result
    else:
        print(f"\n[!] ERROR: Dialogue \"{target_text}\" could not be located in video.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Find exact frame where dialogue appears in a video URL.")
    parser.add_argument("--url", type=str, default="https://ok.ru/video/248244667877", help="Video URL")
    parser.add_argument("--text", type=str, default="My mind rebels at stagnation", help="Target dialogue text")
    parser.add_argument("--output", type=str, default="output_frame.png", help="Output frame image path")
    parser.add_argument("--transcript", type=str, default="transcript.txt", help="Output transcript file path")
    
    args = parser.parse_args()
    find_dialogue_frame(args.url, args.text, args.output, transcript_txt=args.transcript)


if __name__ == "__main__":
    main()
