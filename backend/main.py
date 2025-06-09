from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from detect_trains import run_detection as process_video
import shutil
import os
import pathlib
import json


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

origins = ["*"] 


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/upload")
async def upload_video(video: UploadFile = File(...), threshold: float = Form(...)):
    video_path = os.path.join("uploads", video.filename)
    os.makedirs("uploads", exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        results = process_video(pathlib.Path(video_path), threshold)
        print("Results being sent to frontend:", json.dumps(results, indent=2))
        return {
            "message": "Video processed successfully.",
            "video_path": video_path,
            "statistics": results["statistics"],
            "schedule": results["schedule"]
        }
    except Exception as e:
        return {
            "message": f"Processing failed: {str(e)}"
        }
        
@app.get("/model_metrics")
def get_model_metrics():
    return {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.85,
        "f1_score": 0.87
    }
