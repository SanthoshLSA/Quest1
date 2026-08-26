"""
Module: downloader.py
Handles fetching and downloading media streams from ok.ru, YouTube, direct URLs, and LOCAL FILE PATHS.

Key design decisions:
- If `url` is a local file path, yt-dlp is skipped entirely.
- Cache is URL-aware: new URL = delete old file + fresh download.
- After download, file is re-muxed to seekable MP4 via ffmpeg (handles DASH/fragmented streams).
- Metadata extracted via ffmpeg stderr parse (imageio_ffmpeg ships ffmpeg, not ffprobe).
"""

import os
import re
import json
import shutil
import subprocess
import cv2
import yt_dlp

CACHE_MANIFEST = "downloads/.cache_manifest.json"


def _get_ffmpeg():
    """Returns the bundled ffmpeg executable path."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _get_video_metadata_ffmpeg(video_path, ffmpeg_exe):
    """
    Parses ffmpeg -i stderr output to extract FPS, duration, resolution.
    Works without ffprobe (imageio_ffmpeg only ships ffmpeg).
    """
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-i", video_path],
            capture_output=True, text=True, timeout=30
        )
        stderr = result.stderr

        dur_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr)
        duration = 0.0
        if dur_match:
            h, m, s = dur_match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)

        fps_match = re.search(r"([\d.]+)\s*fps", stderr)
        fps = float(fps_match.group(1)) if fps_match else 23.98

        res_match = re.search(r"(\d{2,5})x(\d{2,5})", stderr)
        width, height = (-1, -1)
        if res_match:
            width, height = int(res_match.group(1)), int(res_match.group(2))

        total_frames = int(round(duration * fps)) if duration > 0 else -1

        if duration > 0:
            return {"fps": fps, "total_frames": total_frames,
                    "duration_sec": duration, "width": width, "height": height}
    except Exception as e:
        print(f"[!] ffmpeg metadata parse failed: {e}")
    return None


def _load_manifest():
    if os.path.exists(CACHE_MANIFEST):
        try:
            with open(CACHE_MANIFEST, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_manifest(data):
    with open(CACHE_MANIFEST, "w") as f:
        json.dump(data, f)


def _remux_to_seekable_mp4(src_path, dst_path, ffmpeg_exe):
    """Re-muxes src to a fully seekable, non-fragmented MP4 at dst."""
    print(f"[*] Re-muxing to seekable MP4: {os.path.basename(src_path)} -> {os.path.basename(dst_path)}")
    cmd = [ffmpeg_exe, "-y", "-i", src_path, "-c", "copy",
           "-movflags", "+faststart", dst_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Re-mux warning (stderr tail): {result.stderr[-300:]}")
    else:
        print(f"[+] Re-mux complete.")
    return os.path.isfile(dst_path) and os.path.getsize(dst_path) > 10000


class MediaDownloader:
    def __init__(self, download_dir="downloads"):
        self.download_dir = os.path.abspath(download_dir)
        os.makedirs(self.download_dir, exist_ok=True)

    def download_media(self, url, progress_callback=None):
        """
        If `url` is a local file path → use it directly (skip yt-dlp).
        Otherwise → download via yt-dlp, then re-mux to seekable MP4.
        Returns metadata dict for the pipeline.
        """
        raw_output = os.path.join(self.download_dir, "input_raw.mp4")
        final_output = os.path.join(self.download_dir, "input_video.mp4")
        ffmpeg_exe = _get_ffmpeg()

        # ── Branch 1: Local file path ─────────────────────────────────────────
        if os.path.isfile(url):
            print(f"[+] Local file detected, skipping download: {url}")
            src_file = url
            if os.path.abspath(src_file) != os.path.abspath(final_output):
                ok = _remux_to_seekable_mp4(src_file, final_output, ffmpeg_exe)
                if not ok:
                    shutil.copy2(src_file, final_output)
                    print(f"[+] Copied local file to: {final_output}")
            _save_manifest({"url": url, "path": final_output})

        # ── Branch 2: Remote URL ──────────────────────────────────────────────
        else:
            manifest = _load_manifest()
            cached_url = manifest.get("url")
            cached_path = manifest.get("path")

            use_cache = False  # Always clean download fresh input stream to avoid low-res stale cache

            if use_cache:
                final_output = cached_path
                print(f"[+] Cache hit. Using: {final_output} "
                      f"({os.path.getsize(final_output)/(1024*1024):.1f} MB)")
            else:
                if cached_url and cached_url != url:
                    print(f"[*] New URL. Removing old cache: {cached_url}")
                    for old in [cached_path, raw_output, final_output]:
                        if old and os.path.isfile(old):
                            try:
                                os.remove(old)
                                print(f"[*] Deleted: {old}")
                            except Exception as e:
                                print(f"[!] Could not delete {old}: {e}")

                # Clean leftover .part files
                for fname in os.listdir(self.download_dir):
                    if fname.endswith((".part", ".ytdl")):
                        try:
                            os.remove(os.path.join(self.download_dir, fname))
                        except Exception:
                            pass

                print(f"[*] Downloading via yt-dlp: {url}")
                last_pct = -1
                last_reported_mb = -1.0

                def ytdlp_progress_hook(d):
                    nonlocal last_pct, last_reported_mb
                    if progress_callback and d.get('status') == 'downloading':
                        total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                        downloaded = d.get('downloaded_bytes', 0)
                        mb_done = downloaded / (1024 * 1024)
                        if total > 0:
                            pct = int(10 + (downloaded / total) * 15) # 10% to 25% range
                            pct = min(25, pct)
                            if pct != last_pct:
                                last_pct = pct
                                mb_total = total / (1024 * 1024)
                                progress_callback(pct, f"Downloading media stream: {mb_done:.1f}MB / {mb_total:.1f}MB ({pct}%)")
                        else:
                            if mb_done - last_reported_mb >= 0.5:
                                last_reported_mb = mb_done
                                progress_callback(15, f"Downloading media stream: {mb_done:.1f}MB (unknown total)")

                ydl_opts = {
                    'format': 'worstvideo[ext=mp4]+worstaudio[ext=m4a]/18/worst[ext=mp4]/worst',
                    'outtmpl': raw_output,
                    'noplaylist': True,
                    'overwrites': True,
                    'quiet': True,
                    'no_warnings': True,
                    'nocheckcertificate': True,
                    'legacy_server_connect': True,
                    'merge_output_format': 'mp4',
                    'progress_hooks': [ytdlp_progress_hook],
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                        'Accept-Language': 'en-US,en;q=0.9'
                    },
                }

                if ffmpeg_exe:
                    ydl_opts['ffmpeg_location'] = ffmpeg_exe

                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.extract_info(url, download=True)
                    print(f"[+] yt-dlp download complete.")
                except Exception as e:
                    print(f"[!] yt-dlp initial attempt notice: {e}")
                    if "ok.ru" in url or "Odnoklassniki" in str(e) or "SSL" in str(e):
                        print("[*] Retrying ok.ru stream ingestion with direct desktop headers...")
                        ydl_opts['http_headers']['Referer'] = 'https://ok.ru/'
                        ydl_opts['check_formats'] = False
                        try:
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                ydl.extract_info(url, download=True)
                        except Exception as e2:
                            print(f"[!] yt-dlp fallback warning: {e2}")

                # Find downloaded file (yt-dlp may rename it)
                src_file = raw_output
                if not os.path.isfile(src_file):
                    for fname in sorted(os.listdir(self.download_dir)):
                        fpath = os.path.join(self.download_dir, fname)
                        if (fname.endswith(('.mp4', '.webm', '.mkv', '.avi'))
                                and not fname.endswith('.part')
                                and os.path.isfile(fpath)
                                and os.path.getsize(fpath) > 1_000_000):
                            src_file = fpath
                            break

                if os.path.isfile(src_file):
                    if os.path.abspath(src_file) == os.path.abspath(final_output):
                        tmp = final_output + ".tmp.mp4"
                        if _remux_to_seekable_mp4(src_file, tmp, ffmpeg_exe):
                            os.replace(tmp, final_output)
                        elif os.path.isfile(tmp):
                            os.remove(tmp)
                    else:
                        _remux_to_seekable_mp4(src_file, final_output, ffmpeg_exe)
                        try:
                            os.remove(src_file)
                        except Exception:
                            pass
                else:
                    print(f"[!] No video file found after download.")

                _save_manifest({"url": url, "path": final_output})

        # ── Metadata extraction ───────────────────────────────────────────────
        meta = _get_video_metadata_ffmpeg(final_output, ffmpeg_exe)

        if meta and meta["fps"] > 0 and meta["total_frames"] > 0:
            fps = meta["fps"]
            total_frames = meta["total_frames"]
            duration = meta["duration_sec"]
            width = meta["width"]
            height = meta["height"]
        else:
            # Fallback: OpenCV
            cap = cv2.VideoCapture(final_output)
            fps = cap.get(cv2.CAP_PROP_FPS) or 23.98
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

        metadata = {
            "title": "Media Stream",
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "resolution": f"{width}x{height}",
            "video_path": final_output,
            "audio_path": final_output,
        }

        print(f"[+] Video Stream Metadata Extracted:")
        print(f"    - FPS        : {metadata['fps']:.2f}")
        print(f"    - Total Frame: {metadata['total_frames']}")
        print(f"    - Duration   : {metadata['duration_sec']:.2f} seconds")
        print(f"    - Resolution : {metadata['resolution']}\n")

        return metadata

    def download_high_quality_clip(self, url, start_sec, end_sec, output_clip_path="output_clip.mp4"):
        """
        Downloads max-resolution 6-second video slice (3s before, 3s after) directly via yt-dlp stream extraction,
        or trims the local input file using ffmpeg re-encoding.
        """
        ffmpeg_exe = _get_ffmpeg()
        os.makedirs(self.download_dir, exist_ok=True)

        # Clean up any existing output_clip files and name variants
        parent_dir = os.path.dirname(os.path.abspath(output_clip_path))
        base_name = os.path.basename(output_clip_path)
        for fname in os.listdir(parent_dir):
            if fname.startswith(base_name) and os.path.isfile(os.path.join(parent_dir, fname)):
                try:
                    os.remove(os.path.join(parent_dir, fname))
                except Exception:
                    pass

        if os.path.isfile(url):
            print(f"[*] Trimming 6s clip from local file: {url}")
            cmd = [
                ffmpeg_exe, "-y",
                "-ss", str(max(0, start_sec)),
                "-to", str(end_sec),
                "-i", url,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                output_clip_path
            ]
            subprocess.run(cmd, capture_output=True, text=True)
            return output_clip_path

        # Setup PATH for yt-dlp ffmpeg execution
        bin_dir = os.path.dirname(ffmpeg_exe)
        exe_alias = os.path.join(bin_dir, "ffmpeg.exe")
        if not os.path.exists(exe_alias):
            try:
                import shutil
                shutil.copy2(ffmpeg_exe, exe_alias)
            except Exception:
                pass
        os.environ["PATH"] = bin_dir + os.path.pathsep + os.environ.get("PATH", "")

        def sec_to_ts(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = s % 60
            return f"{h:02d}:{m:02d}:{sec:06.3f}"

        start_ts = sec_to_ts(max(0, start_sec))
        end_ts = sec_to_ts(end_sec)

        print(f"[*] Downloading maximum resolution 6s clip ({start_ts} - {end_ts}) via yt-dlp...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        if "ok.ru" in url:
            headers['Referer'] = 'https://ok.ru/'

        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': output_clip_path,
            'overwrites': True,
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
            'legacy_server_connect': True,
            'ffmpeg_location': bin_dir,
            'download_ranges': yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)]),
            'force_keyframes_at_cuts': True,
            'http_headers': headers,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
            print(f"[+] yt-dlp download finished.")
        except Exception as e:
            print(f"[!] Partial download notice: {e}")

        # Check if the output file exists at the exact path
        if not os.path.isfile(output_clip_path):
            # Check for name variants (like output_clip.mp4.webm, output_clip.mp4.mkv, etc.)
            candidates = []
            for entry in os.listdir(parent_dir):
                if entry.startswith(base_name) and entry != base_name:
                    full_p = os.path.join(parent_dir, entry)
                    if os.path.isfile(full_p) and not entry.endswith('.part') and not entry.endswith('.ytdl'):
                        candidates.append(full_p)
            
            if candidates:
                downloaded_file = max(candidates, key=os.path.getsize)
                print(f"[+] Found downloaded high-quality clip variant: {downloaded_file}")
                print(f"[*] Re-encoding high-quality clip to H.264/AAC MP4 for browser display & frame extraction...")
                cmd = [
                    ffmpeg_exe, "-y",
                    "-i", downloaded_file,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "fast",
                    "-crf", "20",
                    "-c:a", "aac",
                    output_clip_path
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and os.path.exists(output_clip_path):
                    print(f"[+] Re-encoded successfully to {output_clip_path}")
                    try:
                        os.remove(downloaded_file)
                    except Exception:
                        pass
                else:
                    print(f"[!] Re-encoding failed: {res.stderr[-200:]}. Falling back to copying.")
                    shutil.copy2(downloaded_file, output_clip_path)

        # Fallback to local trim if no high quality clip was generated
        if not os.path.exists(output_clip_path):
            print(f"[!] High-quality clip not found. Trimming from local low-res file...")
            input_vid = os.path.join(self.download_dir, "input_video.mp4")
            if os.path.exists(input_vid):
                cmd = [
                    ffmpeg_exe, "-y",
                    "-ss", str(max(0, start_sec)),
                    "-to", str(end_sec),
                    "-i", input_vid,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "fast",
                    "-crf", "18",
                    "-c:a", "aac",
                    output_clip_path
                ]
                subprocess.run(cmd, capture_output=True, text=True)

        return output_clip_path


if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://ok.ru/video/248244667877"
    downloader = MediaDownloader()
    downloader.download_media(test_url)
