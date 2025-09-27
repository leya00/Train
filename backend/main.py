from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from detect_trains import run_detection as process_video
import shutil
import os
import pathlib
import json
import glob


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
        
        # Debug: Check what's in the results
        print(f"Results keys: {list(results.keys())}")
        print(f"Schedule in results: {results.get('schedule')}")
        print(f"Schedule type: {type(results.get('schedule'))}")
        print(f"Schedule length: {len(results.get('schedule', []))}")
        
        # Check if schedule exists and has content
        if 'schedule' not in results:
            print("ERROR: 'schedule' key missing from results!")
        elif results['schedule'] is None:
            print("ERROR: 'schedule' is None in results!")
        elif len(results['schedule']) == 0:
            print("WARNING: 'schedule' is empty array in results!")
        else:
            print(f"Schedule looks good: {len(results['schedule'])} entries")
        
        response_data = {
            "message": "Video processed successfully.",
            "video_path": video_path,
            "video_name": results.get("video_name", "Unknown"),
            "statistics": results["statistics"],
            "schedule": results["schedule"],
            "processing_timestamp": results.get("processing_timestamp", ""),
            "unique_id": results.get("unique_id", "")
        }
        
        print(f"Response data being sent: {json.dumps(response_data, indent=2)}")
        print(f"Final schedule in response: {response_data.get('schedule')}")
        print(f"Final schedule type: {type(response_data.get('schedule'))}")
        print(f"Final schedule length: {len(response_data.get('schedule', []))}")
        
        return response_data
    except Exception as e:
        return {
            "message": f"Processing failed: {str(e)}",
            "error": str(e)
        }

@app.get("/latest_results")
def get_latest_results():
    """Get the most recent detection results"""
    try:
        results_dir = pathlib.Path("results")
        if not results_dir.exists():
            return {"message": "No results directory found"}
        
        # Get all result files and sort by modification time
        result_files = list(results_dir.glob("detection_results_*.json"))
        if not result_files:
            # Fall back to standard results file
            standard_results = results_dir / "detection_results.json"
            if standard_results.exists():
                with open(standard_results, "r") as f:
                    return json.load(f)
            return {"message": "No results found"}
        
        # Get the most recent file
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, "r") as f:
            return json.load(f)
            
    except Exception as e:
        return {"message": f"Error retrieving results: {str(e)}"}

@app.get("/model_metrics")
def get_model_metrics():
    """Get the latest model metrics from the most recent detection"""
    try:
        results_dir = pathlib.Path("results")
        if not results_dir.exists():
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        # Try to get the latest results
        result_files = list(results_dir.glob("detection_results_*.json"))
        if result_files:
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, "r") as f:
                results = json.load(f)
                stats = results.get("statistics", {})
                return {
                    "accuracy": stats.get("detection_rate", 0.0),
                    "precision": stats.get("precision", 0.0),
                    "recall": stats.get("recall", 0.0),
                    "f1_score": stats.get("f1_score", 0.0)
                }
        
        # Fall back to standard results file
        standard_results = results_dir / "detection_results.json"
        if standard_results.exists():
            with open(standard_results, "r") as f:
                results = json.load(f)
                stats = results.get("statistics", {})
                return {
                    "accuracy": stats.get("detection_rate", 0.0),
                    "precision": stats.get("precision", 0.0),
                    "recall": stats.get("recall", 0.0),
                    "f1_score": stats.get("f1_score", 0.0)
                }
        
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
    except Exception as e:
        print(f"Error getting model metrics: {e}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
