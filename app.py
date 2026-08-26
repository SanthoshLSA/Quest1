"""
FastAPI Local Web Application Server for Dialogue Frame Finder
Runs local server at http://localhost:8000
Streams real-time progress updates to the frontend via Server-Sent Events (SSE).
"""

import os
import json
import queue
import threading
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from main import find_dialogue_frame

app = FastAPI(title="Dialogue Frame Finder Web App")

# Serve static web app files
app.mount("/static", StaticFiles(directory="static"), name="static")


class SearchRequest(BaseModel):
    url: str
    text: str
    skip_ocr: bool = False
    model_size: str = "small"


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.get("/output_frame.png")
def get_output_frame():
    if os.path.exists("output_frame.png"):
        return FileResponse("output_frame.png", media_type="image/png")
    raise HTTPException(status_code=404, detail="Frame image not generated yet.")


@app.get("/output_clip.mp4")
def get_output_clip():
    if os.path.exists("output_clip.mp4"):
        return FileResponse("output_clip.mp4", media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video clip not generated yet.")


@app.post("/api/find_dialogue_stream")
def api_find_dialogue_stream(req: SearchRequest):
    """
    SSE endpoint: streams real-time progress events to the frontend,
    then sends the final result or error as the last event.
    """
    event_queue: queue.Queue = queue.Queue()

    def progress_callback(percent: int, msg: str):
        event_queue.put({"type": "progress", "percent": percent, "message": msg})

    def run_pipeline():
        try:
            res = find_dialogue_frame(
                video_url=req.url,
                target_text=req.text,
                output_img="output_frame.png",
                transcript_txt="transcript.txt",
                transcript_json="transcript.json",
                progress_callback=progress_callback,
                skip_ocr=req.skip_ocr,
                model_size=req.model_size,
            )

            # Download failed — surface error immediately, skip transcript loading
            if res.get("download_error"):
                event_queue.put({"type": "error", "error": res["error_message"]})
                return

            transcript_segments = []
            if os.path.exists("transcript.json"):
                with open("transcript.json", "r", encoding="utf-8") as f:
                    tdata = json.load(f)
                    transcript_segments = tdata.get("segments", [])

            if res.get("found", True):
                event_queue.put({
                    "type": "result",
                    "result": {
                        "timestamp": res["timestamp"],
                        "frame": res["frame"],
                        "text": res["text"],
                        "total_frames": res.get("total_frames", 0),
                        "fps": res.get("fps", 0),
                        "image_url": "/output_frame.png",
                        "clip_url": "/output_clip.mp4",
                        "transcript_segments": transcript_segments,
                    }
                })
            else:
                event_queue.put({
                    "type": "not_found",
                    "result": {
                        "transcript_segments": transcript_segments,
                    }
                })
        except Exception as e:
            print(f"[!] SSE Pipeline Error: {e}")
            event_queue.put({"type": "error", "error": str(e)})
        finally:
            event_queue.put(None)  # Sentinel: signals stream end

    # Run heavy pipeline in a background thread so we can yield SSE events
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    def event_generator():
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind proxy
        },
    )


if __name__ == "__main__":
    import uvicorn
    print("[*] Starting Dialogue Frame Finder at http://localhost:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
