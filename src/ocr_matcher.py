"""
Module: ocr_matcher.py
Uses EasyOCR and OpenCV image preprocessing to extract on-screen visual text from candidate frames
and pinpoint exact visual entry frames using RapidFuzz string matching.
"""

import cv2
import easyocr
import numpy as np
from rapidfuzz import fuzz

class VisualOCRMatcher:
    def __init__(self, languages=['en'], gpu=False):
        print(f"[*] Initializing EasyOCR Engine (languages={languages}, gpu={gpu})...")
        # verbose=False suppresses ASCII progress bar unicode error on Windows
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)

    def preprocess_frame(self, frame):
        """
        Applies grayscale conversion and CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to enhance subtitle text readability on compressed video frames.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def extract_text_from_frame(self, frame):
        """
        Runs EasyOCR on a single OpenCV BGR frame array.
        Returns combined extracted text string and bounding box details.
        """
        enhanced_frame = self.preprocess_frame(frame)
        results = self.reader.readtext(enhanced_frame)
        
        extracted_texts = []
        for bbox, text, prob in results:
            if prob >= 0.2:
                extracted_texts.append(text)
                
        combined_text = " ".join(extracted_texts)
        return combined_text, results

    def inspect_frame_window(self, video_path, target_text, start_frame, end_frame, fps=23.98):
        """
        Scans a candidate frame window [start_frame, end_frame] frame-by-frame
        to find the exact first frame index where on-screen text matching target_text appears.
        """
        print(f"[*] Inspecting visual frame window [{start_frame} to {end_frame}] via EasyOCR...")
        cap = cv2.VideoCapture(video_path)
        
        target_clean = target_text.lower().strip()
        first_match = None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame_idx = start_frame
        
        while curr_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            combined_text, _ = self.extract_text_from_frame(frame)
            if combined_text:
                score = fuzz.partial_ratio(target_clean, combined_text.lower())
                if score >= 65.0:
                    timestamp_sec = curr_frame_idx / fps
                    hrs = int(timestamp_sec // 3600)
                    mins = int((timestamp_sec % 3600) // 60)
                    secs = timestamp_sec % 60
                    timestamp_str = f"{hrs:02d}:{mins:02d}:{secs:06.3f}"

                    first_match = {
                        "found": True,
                        "frame": curr_frame_idx,
                        "timestamp": timestamp_str,
                        "text": combined_text,
                        "score": score,
                        "frame_img": frame
                    }
                    print(f"[+] Exact Visual Text Match Found on Screen!")
                    print(f"    - Frame Number : {curr_frame_idx}")
                    print(f"    - Timestamp    : {timestamp_str}")
                    print(f"    - OCR Text     : \"{combined_text}\"")
                    print(f"    - Match Score  : {score:.1f}%")
                    break

            curr_frame_idx += 1

        cap.release()

        if not first_match:
            print(f"[!] No visual subtitle text matched target in window [{start_frame}-{end_frame}].")
            return {"found": False}
            
        return first_match


if __name__ == "__main__":
    import os
    video_file = "downloads/input_video.mp4"
    if os.path.exists(video_file):
        matcher = VisualOCRMatcher()
        matcher.inspect_frame_window(video_file, "My mind rebels at stagnation", start_frame=7750, end_frame=7850)
    else:
        print("[!] Video file not found.")
