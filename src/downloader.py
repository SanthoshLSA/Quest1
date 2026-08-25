"""
Module: downloader.py
Handles fetching and downloading media streams from ok.ru, YouTube, and direct video URLs using yt-dlp.
"""

import os
import cv2
import yt_dlp

class MediaDownloader:
    def __init__(self, download_dir="downloads"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def download_media(self, url):
        """
        Downloads video and extracts audio using yt-dlp.
        Returns a dict containing paths to local video and audio files, plus metadata.
        """
        print(f"[*] Ingesting video URL via yt-dlp: {url}")
        
        video_output = os.path.join(self.download_dir, "input_video.mp4")
        audio_output = os.path.join(self.download_dir, "input_audio.wav")
        
        # Configure yt-dlp options for best quality mp4 video and wav audio
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': video_output,
            'overwrites': True,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': []
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            
        print(f"[+] Video successfully downloaded to: {video_output}")

        # Extract audio using ffmpeg or OpenCV/moviepy if available, or yt-dlp audio extractor
        audio_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.download_dir, "input_audio.%(ext)s"),
            'overwrites': True,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }]
        }
        
        try:
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.extract_info(url, download=True)
            if not os.path.exists(audio_output):
                # Fallback check for audio file extension
                for f in os.listdir(self.download_dir):
                    if f.startswith("input_audio"):
                        audio_output = os.path.join(self.download_dir, f)
                        break
        except Exception as e:
            print(f"[!] Audio extraction warning: {e}. Will rely on video track directly.")
            audio_output = None

        # Fetch metadata using OpenCV
        cap = cv2.VideoCapture(video_output)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        metadata = {
            "title": info_dict.get("title", "Unknown Video"),
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "resolution": f"{width}x{height}",
            "video_path": video_output,
            "audio_path": audio_output
        }

        print(f"[+] Video Metadata Extracted:")
        print(f"    - Title      : {metadata['title']}")
        print(f"    - FPS        : {metadata['fps']:.2f}")
        print(f"    - Total Frame: {metadata['total_frames']}")
        print(f"    - Duration   : {metadata['duration_sec']:.2f} seconds")
        print(f"    - Resolution : {metadata['resolution']}")

        return metadata


if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://ok.ru/video/248244667877"
    downloader = MediaDownloader()
    downloader.download_media(test_url)
