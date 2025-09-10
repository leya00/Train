import os
import sys
import json
import pathlib
import argparse
import uuid
from datetime import timedelta
import numpy as np
from ultralytics import YOLO
import cv2


def extract_frames(video_path, output_folder, fps_interval=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid video properties: fps={fps}, frame_count={frame_count}")
    
    duration_seconds = frame_count / fps
    frame_interval = max(1, int(fps * fps_interval))  # Ensure at least 1 frame interval
    count, saved = 0, 0
    frame_timestamps = {}

    print(f"Video properties: fps={fps}, total_frames={frame_count}, duration={duration_seconds:.2f}s")
    print(f"Extracting frames every {frame_interval} frames (every {fps_interval:.1f} seconds)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"frame_{saved:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            success = cv2.imwrite(frame_path, frame)
            if not success:
                print(f"Warning: Failed to save frame {frame_name}")
                continue
                
            timestamp_seconds = count / fps
            timestamp = str(timedelta(seconds=int(timestamp_seconds)))
            frame_timestamps[frame_name] = {
                "timestamp": timestamp,
                "timestamp_seconds": timestamp_seconds,
            }
            saved += 1
        count += 1

    cap.release()
    
    if saved == 0:
        raise ValueError("No frames were extracted from the video")
    
    print(f"Successfully extracted {saved} frames from {count} total frames")
    return frame_timestamps, duration_seconds


def generate_train_schedule(duration_seconds, num_trains=5):
    print(f"Generating schedule for duration: {duration_seconds} seconds")
    
    # Ensure we have a minimum duration for schedule generation
    if duration_seconds < 1:
        print(f"Warning: Very short duration ({duration_seconds}s), setting minimum duration")
        duration_seconds = 1
    
    # Ensure we have at least one train in the schedule
    if num_trains < 1:
        print(f"Warning: Invalid num_trains ({num_trains}), setting to 1")
        num_trains = 1
    
    interval = duration_seconds / (num_trains + 1)
    print(f"Schedule interval: {interval} seconds")
    
    schedule = []
    for i in range(num_trains):
        expected_time = interval * (i + 1)
        schedule_entry = {
            "train_id": i + 1,
            "expected_time": expected_time,
            "expected_timestamp": str(timedelta(seconds=int(expected_time))),
            "tolerance": 15,
            "detected": False,
            "detection_times": [],
        }
        schedule.append(schedule_entry)
        print(f"  Train {i+1}: Expected at {expected_time:.1f}s ({schedule_entry['expected_timestamp']})")
    
    print(f"Generated {len(schedule)} schedule entries")
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
    # Validate inputs to prevent division by zero
    if total_frames <= 0:
        print("Warning: total_frames is 0 or negative, setting to 1 to avoid division by zero")
        total_frames = 1
    
    frames_with_trains = len(train_frames)
    detection_rate = round(frames_with_trains / total_frames * 100, 2) if total_frames > 0 else 0

    confidences = []
    object_counts = {}  # Dictionary to store counts of each detected object
    false_positives = 0
    true_positives = 0

    for frame in train_frames:
        for det in frame["detections"]:
            confidences.append(det["confidence"])
            # Count objects by class
            class_name = det["class"]
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            # For train detection, we consider detections above threshold as potential true positives
            # and detections below threshold as potential false positives
            if det["confidence"] >= threshold:
                true_positives += 1
            else:
                # Low confidence detections might be false positives
                false_positives += 1

    avg_confidence = round(np.mean(confidences), 2) if confidences else 0
    duration_hours = duration_seconds / 3600
    trains_per_hour = round(frames_with_trains / duration_hours, 1) if duration_hours > 0 else 0

    # Calculate precision: True Positives / (True Positives + False Positives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Calculate recall dynamically based on schedule detections
    # Recall = True Positives / (True Positives + False Negatives)
    # For our case: detected trains / expected trains from schedule
    expected_trains = len(schedule) if schedule else 1
    detected_trains = sum(1 for entry in schedule if entry.get("detected", False))
    recall = detected_trains / expected_trains if expected_trains > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total_frames": total_frames,
        "frames_with_trains": frames_with_trains,
        "false_positives": false_positives,
        "true_positives": true_positives,
        "detection_rate": detection_rate,
        "avg_confidence": avg_confidence,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "duration_seconds": duration_seconds,
        "object_counts": object_counts,  # Add object counts to the statistics
        "expected_trains": expected_trains,
        "detected_trains": detected_trains
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

        # Clear previous frames and results to avoid contamination
        if input_folder.exists():
            for old_file in input_folder.glob("*.jpg"):
                old_file.unlink()
            print("Cleared previous frames")
        
        if output_folder.exists():
            for old_file in output_folder.glob("*.jpg"):
                old_file.unlink()
            print("Cleared previous inference results")

        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        print(f"Extracting frames from video: {video_path}")
        frame_timestamps, duration_seconds = extract_frames(video_path, str(input_folder), fps_interval)
        
        if not frame_timestamps:
            raise ValueError("No frames were extracted from the video")
            
        print(f"Extracted {len(frame_timestamps)} frames")

        schedule = generate_train_schedule(duration_seconds)
        print(f"Generated schedule: {json.dumps(schedule, indent=2)}")
        print(f"Schedule length: {len(schedule)}")

        print("Loading YOLO model...")
        try:
            # Try to load with newer PyTorch compatibility
            import torch
            import torch.serialization
            
            # For PyTorch 2.6+ with weights_only=True, we need to add safe globals
            if hasattr(torch.serialization, 'add_safe_globals'):
                try:
                    # Add all the necessary safe globals for YOLO model loading
                    from ultralytics.nn.tasks import DetectionModel
                    from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
                    from ultralytics.nn.modules.block import DFL, Proto
                    from ultralytics.nn.modules.conv import Conv, ConvTranspose
                    from ultralytics.nn.modules.transformer import TransformerBlock
                    from ultralytics.nn.modules.head import Classify, Detect, Pose, RTDETRDecoder, Segment
                    
                    torch.serialization.add_safe_globals([
                        DetectionModel, Conv, C2f, SPPF, Detect, DFL, Proto,
                        ConvTranspose, TransformerBlock, Classify, Pose, RTDETRDecoder, Segment
                    ])
                    print("Added safe globals for PyTorch 2.6+ compatibility")
                except ImportError as e:
                    print(f"Warning: Could not import some YOLO modules: {e}")
                
                # Try loading with weights_only=True first
                try:
                    model = YOLO("yolov8n.pt")
                    print("Model loaded successfully with weights_only=True")
                    model_loaded = True
                except Exception as e:
                    print(f"weights_only=True failed, trying with weights_only=False: {e}")
                    # Fallback to weights_only=False (less secure but should work)
                    try:
                        # Temporarily set torch.load to use weights_only=False
                        original_load = torch.load
                        def safe_load(*args, **kwargs):
                            kwargs['weights_only'] = False
                            return original_load(*args, **kwargs)
                        
                        # Monkey patch torch.load temporarily
                        torch.load = safe_load
                        model = YOLO("yolov8n.pt")
                        # Restore original torch.load
                        torch.load = original_load
                        print("Model loaded successfully with weights_only=False")
                        model_loaded = True
                    except Exception as e2:
                        print(f"Both loading methods failed: {e2}")
                        model_loaded = False
                        model = None
            else:
                # For older PyTorch versions
                model = YOLO("yolov8n.pt")
                print("Model loaded successfully with older PyTorch")
                model_loaded = True
                
        except Exception as model_error:
            print(f"Warning: Failed to load YOLO model: {model_error}")
            print("Continuing with schedule generation only...")
            model_loaded = False
            model = None

        train_frames = []

        if model_loaded:
            print("Running inference on frames...")
            for img in sorted(os.listdir(input_folder)):
                if not img.endswith('.jpg'):
                    continue
                    
                frame_path = input_folder / img
                try:
                    result = model(frame_path)

                    train_objects = []
                    for box in result[0].boxes:
                        cls_id = int(box.cls)
                        cls_name = result[0].names[cls_id]
                        conf = float(box.conf)
                        # Save ALL train detections regardless of confidence
                        # The metrics function will determine true vs false positives
                        if cls_name.lower() == "train" or cls_id == 6:
                            train_objects.append({
                                "class": cls_name,
                                "confidence": round(conf, 2),
                                "bbox": box.xyxy[0].tolist()
                            })

                    # Save frame if it has ANY train detections (high or low confidence)
                    if train_objects and img in frame_timestamps:
                        train_frames.append({
                            "frame": img,
                            "timestamp": frame_timestamps[img]["timestamp"],
                            "timestamp_seconds": frame_timestamps[img]["timestamp_seconds"],
                            "path": str(output_folder / img),
                            "detections": train_objects
                        })
                        
                        # Try to save the annotated image if possible
                        try:
                            # Use the original frame as fallback since result.save() is failing
                            import cv2
                            frame_path = input_folder / img
                            if frame_path.exists():
                                cv2.imwrite(str(output_folder / img), cv2.imread(str(frame_path)))
                        except Exception as save_error:
                            print(f"Warning: Could not save frame {img}: {save_error}")
                        
                except Exception as frame_error:
                    print(f"Warning: Failed to process frame {img}: {frame_error}")
                    continue
        else:
            print("Skipping inference due to model loading failure")
            # Create placeholder train frames for testing
            train_frames = []
                
        end_time = time.time()
        detection_time = round(end_time - start_time, 2)
        num_detected_frames = len(os.listdir(input_folder))
        detection_fps = round(num_detected_frames / detection_time, 2) if detection_time > 0 else 0

        print(f"Train frames found: {len(train_frames)}")
        print(f"Schedule before detection: {len(schedule)} entries")
        
        updated_schedule = check_schedule_detections(train_frames, schedule, threshold)
        print(f"Updated schedule after detection: {json.dumps(updated_schedule, indent=2)}")
        print(f"Schedule after detection: {len(updated_schedule)} entries")
        
        # Ensure we have a valid schedule (fallback if something went wrong)
        if not updated_schedule or len(updated_schedule) == 0:
            print("Warning: Schedule is empty, generating fallback schedule")
            fallback_duration = max(duration_seconds, 10)  # At least 10 seconds
            updated_schedule = generate_train_schedule(fallback_duration, num_trains=3)
            print(f"Fallback schedule generated: {len(updated_schedule)} entries")

        detection_stats = calculate_metrics(
            train_frames=train_frames,
            total_frames=len(frame_timestamps),
            duration_seconds=duration_seconds,
            schedule=updated_schedule,
            threshold=threshold
        )
        
        # If model failed to load, add warning to statistics
        if not model_loaded:
            detection_stats["model_loading_failed"] = True
            detection_stats["warning"] = "YOLO model failed to load - metrics are based on schedule only"
            print("Warning: Using fallback statistics due to model loading failure")
        
        detection_stats["detection_time_seconds"] = detection_time
        detection_stats["detection_fps"] = detection_fps

        results = {
            "train_frames": train_frames,
            "statistics": detection_stats,
            "video_path": str(video_path),
            "video_name": video_path.stem,
            "schedule": updated_schedule,
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unique_id": str(uuid.uuid4())
        }

        # Validate schedule structure before returning
        if updated_schedule and len(updated_schedule) > 0:
            print(f"Final schedule validation: {len(updated_schedule)} entries")
            for i, entry in enumerate(updated_schedule):
                if not isinstance(entry, dict):
                    print(f"Warning: Schedule entry {i} is not a dict: {entry}")
                    continue
                required_fields = ["train_id", "expected_time", "expected_timestamp", "tolerance", "detected", "detection_times"]
                missing_fields = [field for field in required_fields if field not in entry]
                if missing_fields:
                    print(f"Warning: Schedule entry {i} missing fields: {missing_fields}")
        else:
            print("Warning: Schedule is still empty, this may cause frontend issues")

        print(f"Final results being returned: {json.dumps(results, indent=2)}")

        # Save results with unique filename per video
        video_name = video_path.stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        results_filename = f"detection_results_{video_name}_{timestamp}_{unique_id}.json"
        results_path = results_dir / results_filename
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_path}")
        
        # Also save to the standard filename for backward compatibility
        standard_results_path = results_dir / "detection_results.json"
        with open(standard_results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results also saved to standard location: {standard_results_path}")
        return results

    except Exception as e:
        print(f"Error in run_detection: {e}", file=sys.stderr)
        raise  # Raise the exception instead of sys.exit(1)


def cleanup_old_results(results_dir, keep_recent=5):
    """Clean up old result files, keeping only the most recent ones"""
    try:
        result_files = list(results_dir.glob("detection_results_*.json"))
        if len(result_files) > keep_recent:
            # Sort by modification time and keep only the most recent
            result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            files_to_delete = result_files[keep_recent:]
            for old_file in files_to_delete:
                old_file.unlink()
                print(f"Cleaned up old result file: {old_file.name}")
    except Exception as e:
        print(f"Warning: Could not cleanup old results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train detection script')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--threshold', type=float, default=0.8, help='Detection confidence threshold')
    parser.add_argument('--fps_interval', type=float, default=1, help='Interval (seconds) between extracted frames')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old result files')
    args = parser.parse_args()

    try:
        video_path = pathlib.Path(args.video_path)
        print(f"Starting train detection for video: {video_path}")
        print(f"Confidence threshold: {args.threshold}")
        print(f"Frame extraction interval: {args.fps_interval} seconds")
        
        results = run_detection(video_path, args.threshold, args.fps_interval)
        
        # Cleanup old results if requested
        if args.cleanup:
            BASE_DIR = pathlib.Path(__file__).parent.resolve()
            results_dir = BASE_DIR / "results"
            cleanup_old_results(results_dir)
        
        print("Train detection completed successfully!")
        print(f"Detection rate: {results['statistics']['detection_rate']}%")
        print(f"F1 Score: {results['statistics']['f1_score']}")
        print(f"Precision: {results['statistics']['precision']}")
        print(f"Recall: {results['statistics']['recall']}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
