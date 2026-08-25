"""
FastAPI Local Web Application Server for Dialogue Frame Finder
Runs local server at http://localhost:8000
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from main import find_dialogue_frame

app = FastAPI(title="Dialogue Frame Finder Web App")

# Serve static web app files
app.mount("/static", StaticFiles(directory="static"), name="static")

class SearchRequest(BaseModel):
    url: str
    text: str

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/output_frame.png")
def get_output_frame():
    if os.path.exists("output_frame.png"):
        return FileResponse("output_frame.png")
    raise HTTPException(status_code=404, detail="Frame image not generated yet.")

@app.post("/api/find_dialogue")
def api_find_dialogue(req: SearchRequest):
    print(f"[*] API Received Search Request: URL='{req.url}', Text='{req.text}'")
    try:
        res = find_dialogue_frame(
            video_url=req.url,
            target_text=req.text,
            output_img="output_frame.png",
            transcript_txt="transcript.txt",
            transcript_json="transcript.json"
        )
        
        transcript_segments = []
        if os.path.exists("transcript.json"):
            with open("transcript.json", "r", encoding="utf-8") as f:
                tdata = json.load(f)
                transcript_segments = tdata.get("segments", [])

        return {
            "success": True,
            "result": {
                "timestamp": res["timestamp"],
                "frame": res["frame"],
                "text": res["text"],
                "image_url": "/output_frame.png",
                "transcript_segments": transcript_segments
            }
        }
    except Exception as e:
        print(f"[!] API Execution Error: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("[*] Starting Local Web Application Server at http://localhost:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
