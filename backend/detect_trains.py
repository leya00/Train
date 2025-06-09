import os
import sys
import json
import pathlib
import argparse
from datetime import timedelta
import numpy as np
from ultralytics import YOLO
import cv2


def extract_frames(video_path, output_folder, fps_interval=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = frame_count / fps if fps > 0 else 0
    frame_interval = int(fps * fps_interval)
    count, saved = 0, 0
    frame_timestamps = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"frame_{saved:04d}.jpg"
            cv2.imwrite(f"{output_folder}/{frame_name}", frame)
            timestamp_seconds = count / fps
            timestamp = str(timedelta(seconds=int(timestamp_seconds)))
            frame_timestamps[frame_name] = {
                "timestamp": timestamp,
                "timestamp_seconds": timestamp_seconds,
            }
            saved += 1
        count += 1

    cap.release()
    return frame_timestamps, duration_seconds


def generate_train_schedule(duration_seconds, num_trains=5):
    interval = duration_seconds / (num_trains + 1)
    schedule = []
    for i in range(num_trains):
        expected_time = interval * (i + 1)
        schedule.append({
            "train_id": i + 1,
            "expected_time": expected_time,
            "expected_timestamp": str(timedelta(seconds=int(expected_time))),  # ← NEW
            "tolerance": 15,
            "detected": False,
            "detection_times": [],
        })
    return schedule


def check_schedule_detections(train_frames, schedule, threshold):
    """Check if trains were detected at scheduled times"""
    for entry in schedule:
        expected_time = entry["expected_time"]
        tolerance = entry["tolerance"]
        
        # Find detections within tolerance window
        detections = []
        for frame in train_frames:
            frame_time = frame["timestamp_seconds"]
            if abs(frame_time - expected_time) <= tolerance:
                for detection in frame["detections"]:
                    if detection["class"] == "train" and detection["confidence"] >= threshold:
                        detections.append({
                            "time": frame_time,
                            "confidence": detection["confidence"]
                        })
        
        # Update schedule entry with detection results
        entry["detected"] = len(detections) > 0
        if detections:
            # Sort detections by time and take the closest one to expected time
            detections.sort(key=lambda x: abs(x["time"] - expected_time))
            entry["detection_times"] = [detections[0]["time"]]
            entry["confidence"] = detections[0]["confidence"]
        else:
            entry["detection_times"] = []
            entry["confidence"] = 0.0
    
    return schedule


def calculate_metrics(train_frames, total_frames, duration_seconds, schedule, threshold=0.8):
    frames_with_trains = len(train_frames)
    detection_rate = round(frames_with_trains / total_frames * 100, 2) if total_frames else 0

    confidences = []
    object_counts = {}  # Dictionary to store counts of each detected object
    false_positives = 0

    for frame in train_frames:
        for det in frame["detections"]:
            confidences.append(det["confidence"])
            # Count objects by class
            class_name = det["class"]
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            # Count false positives (detections below threshold)
            if det["confidence"] < threshold:
                false_positives += 1

    avg_confidence = round(np.mean(confidences), 2) if confidences else 0
    duration_hours = duration_seconds / 3600
    trains_per_hour = round(frames_with_trains / duration_hours, 1) if duration_hours > 0 else 0

    precision = frames_with_trains / (frames_with_trains + false_positives) if (frames_with_trains + false_positives) > 0 else 0
    recall = 0.85  #Replace with real recall from ground truth
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total_frames": total_frames,
        "frames_with_trains": frames_with_trains,
        "detection_rate": detection_rate,
        "avg_confidence": avg_confidence,
        "false_positives": false_positives,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "duration_seconds": duration_seconds,
        "object_counts": object_counts  # Add object counts to the statistics
    }


def run_detection(video_path: pathlib.Path, threshold=0.8, fps_interval=1):
    import time
    start_time = time.time()

    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    input_folder = BASE_DIR / "frames"
    output_folder = BASE_DIR / "inference_results"
    results_dir = BASE_DIR / "results"

    try:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path}")

        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        print(f"Extracting frames from video: {video_path}")
        frame_timestamps, duration_seconds = extract_frames(video_path, str(input_folder), fps_interval)
        print(f"Extracted {len(frame_timestamps)} frames")

        schedule = generate_train_schedule(duration_seconds)
        print(f"Generated schedule: {json.dumps(schedule, indent=2)}")

        print("Loading YOLO model...")
        model = YOLO("yolov8n.pt")
        print("Model loaded successfully")

        train_frames = []

        print("Running inference on frames...")
        for img in sorted(os.listdir(input_folder)):
            if not img.endswith('.jpg'):
                continue
                
            frame_path = input_folder / img
            result = model(frame_path)

            train_objects = []
            for box in result[0].boxes:
                cls_id = int(box.cls)
                cls_name = result[0].names[cls_id]
                conf = float(box.conf)
                if (cls_name.lower() == "train" or cls_id == 6) and conf >= threshold:
                    train_objects.append({
                        "class": cls_name,
                        "confidence": round(conf, 2),
                        "bbox": box.xyxy[0].tolist()
                    })

            result[0].save(filename=str(output_folder / img))

            if train_objects and img in frame_timestamps:
                train_frames.append({
                    "frame": img,
                    "timestamp": frame_timestamps[img]["timestamp"],
                    "timestamp_seconds": frame_timestamps[img]["timestamp_seconds"],
                    "path": str(output_folder / img),
                    "detections": train_objects
                })
                
        end_time = time.time()
        detection_time = round(end_time - start_time, 2)
        num_detected_frames = len(os.listdir(input_folder))
        detection_fps = round(num_detected_frames / detection_time, 2) if detection_time > 0 else 0

        updated_schedule = check_schedule_detections(train_frames, schedule, threshold)
        print(f"Updated schedule after detection: {json.dumps(updated_schedule, indent=2)}")

        detection_stats = calculate_metrics(
            train_frames=train_frames,
            total_frames=len(frame_timestamps),
            duration_seconds=duration_seconds,
            schedule=updated_schedule,
            threshold=threshold
        )
        
        detection_stats["detection_time_seconds"] = detection_time
        detection_stats["detection_fps"] = detection_fps

        results = {
            "train_frames": train_frames,
            "statistics": detection_stats,
            "video_path": str(video_path),
            "schedule": updated_schedule,
        }

        print(f"Final results being returned: {json.dumps(results, indent=2)}")

        results_path = results_dir / "detection_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_path}")
        return results

    except Exception as e:
        print(f"Error in run_detection: {e}", file=sys.stderr)
        raise  # Raise the exception instead of sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train detection script')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--threshold', type=float, default=0.8, help='Detection confidence threshold')
    parser.add_argument('--fps_interval', type=float, default=1, help='Interval (seconds) between extracted frames')
    args = parser.parse_args()

    run_detection(pathlib.Path(args.video_path), args.threshold, args.fps_interval)


if __name__ == "__main__":
    main()
