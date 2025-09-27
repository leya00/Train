import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tempfile
import shutil
from pathlib import Path
import time
import json
import glob
from collections import defaultdict
import torch
import sys

app = Flask(__name__)
CORS(app)

# Global variables
model = None
inference_metrics = {
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "mAP50": 0.0,
    "mAP50_95": 0.0,
    "total_detections": 0,
    "files_processed": 0,
    "avg_confidence": 0.0
}

# Create necessary directories
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/inference-results", exist_ok=True)

def initialize_metrics():
    """Reset metrics for each inference run"""
    global inference_metrics
    inference_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "total_detections": 0,
        "files_processed": 0,
        "avg_confidence": 0.0
    }

def load_model():
    """Load the YOLOv5 model from the federated learning output"""
    global model #here
    
    # Try to load the federated model
    model_paths = [
        "static/output/final_model.pt",
        "model/my_model.pt"
    ]
    
    for model_path in model_paths:
        try:
            print(f"Loading model from {model_path}...")
            # Load YOLOv5 model
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            if model:
                print(f"Successfully loaded model: {model_path}")
                return model
        except Exception as e:
            print(f"Failed to load {model_path}: {str(e)}")
    
    # If we couldn't load the federated model, try to load a default YOLOv5 model
    try:
        print("Loading default YOLOv5 model...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        print("Successfully loaded default YOLOv5 model")
        return model
    except Exception as e:
        print(f"Failed to load default YOLOv5 model: {str(e)}")
        return None

def get_federated_metrics():
    """Get metrics from the federated learning model's training results"""
    try:
        # Try to read metrics from results.csv using pandas
        results_path = "model/train/results.csv"
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            if not df.empty:
                # Get the last row (final metrics)
                last_row = df.iloc[-1]
                metrics = {
                    "precision": float(last_row.get("metrics/precision(B)", 0.0)),
                    "recall": float(last_row.get("metrics/recall(B)", 0.0)),
                    "mAP50": float(last_row.get("metrics/mAP50(B)", 0.0)),
                    "mAP50_95": float(last_row.get("metrics/mAP50-95(B)", 0.0)),
                    "box_loss": float(last_row.get("val/box_loss", 0.0)),
                    "cls_loss": float(last_row.get("val/cls_loss", 0.0)),
                    "dfl_loss": float(last_row.get("val/dfl_loss", 0.0)),
                    "epochs": int(last_row.get("epoch", 0)),
                }
                return metrics
    except Exception as e:
        print(f"Error reading metrics from results.csv: {e}")
    
    # Fallback: Return default metrics
    return {
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "box_loss": 0.0,
        "cls_loss": 0.0,
        "dfl_loss": 0.0,
        "epochs": 0,
    }

def run_inference(file_path, is_video=False):
    """Run inference using YOLOv5 model"""
    global model
    
    if model is None:
        return None, "Model not loaded"
    
    try:
        start_time = time.time()
        
        if is_video:
            # For video, we need to process frame by frame
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return None, "Failed to open video file"
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create output video writer
            output_filename = os.path.basename(file_path)
            output_path = f"static/inference-results/processed_{output_filename}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            detections = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference on frame
                results = model(frame)
                
                # Get detections
                pred = results.pred[0]
                if pred is not None and len(pred) > 0:
                    for *xyxy, conf, cls in pred:
                        # Add detection to list
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        confidence = float(conf)
                        class_id = int(cls)
                        class_name = results.names[class_id]
                        
                        detections.append({
                            "frame": frame_count,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class": class_id,
                            "class_name": class_name
                        })
                
                # Draw results on frame
                annotated_frame = results.render()[0]
                
                # Write frame to output video
                out.write(annotated_frame)
                
                frame_count += 1
            
            cap.release()
            out.release()
            
        else:
            # For image
            results = model(file_path)
            
            # Save the annotated image
            output_filename = os.path.basename(file_path)
            output_path = f"static/inference-results/processed_{output_filename}"
            results.save(save_dir="static/inference-results", exist_ok=True)
            
            # Get detections
            detections = []
            pred = results.pred[0]
            if pred is not None and len(pred) > 0:
                for *xyxy, conf, cls in pred:
                    # Add detection to list
                    x1, y1, x2, y2 = [int(x) for x in xyxy]
                    confidence = float(conf)
                    class_id = int(cls)
                    class_name = results.names[class_id]
                    
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class": class_id,
                        "class_name": class_name
                    })
        
        processing_time = time.time() - start_time
        
        # Calculate metrics based on confidence threshold
        true_positives = sum(1 for d in detections if d["confidence"] > 0.7)
        false_positives = sum(1 for d in detections if d["confidence"] <= 0.7)
        false_negatives = 0  # We don't have ground truth, so we assume 0
        
        # Calculate precision, recall, F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "output_path": f"/inference-results/processed_{output_filename}",
            "detections": detections,
            "processing_time": processing_time,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }, None
        
    except Exception as e:
        return None, f"Error during inference: {str(e)}"

def get_model_info():
    """Get information about the model and metrics"""
    # Get federated model metrics from training
    federated_metrics = get_federated_metrics()
    
    # Get the current dynamic inference metrics
    global inference_metrics
    
    # Combine metrics
    metrics = {
        "model_name": "Federated Learning YOLOv5 Model",
        "model_type": "Object Detection",
        "model_status": "Loaded" if model is not None else "Not Loaded",
        "federated_metrics": federated_metrics,
    }
    
    # Add inference metrics if available
    if inference_metrics["files_processed"] > 0:
        metrics.update({
            "batch_metrics": {
                "precision": inference_metrics["precision"],
                "recall": inference_metrics["recall"],
                "f1_score": inference_metrics["f1_score"],
                "files_processed": inference_metrics["files_processed"],
                "total_detections": inference_metrics["total_detections"],
                "avg_confidence": inference_metrics["avg_confidence"]
            }
        })
    else:
        metrics.update({
            "batch_metrics": {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "files_processed": 0,
                "total_detections": 0,
                "avg_confidence": 0.0
            }
        })
    
    return metrics

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    """Return model information and metrics"""
    info = get_model_info()
    return jsonify(info)

@app.route('/run-inference', methods=['POST'])
def run_inference_endpoint():
    """Run inference on uploaded files"""
    global inference_metrics
    
    # Reset metrics for this batch
    initialize_metrics()
    
    # Process uploaded files
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No files uploaded"}), 400
    
    results = []
    errors = []
    total_detections = 0
    total_confidence = 0
    
    # Batch metrics
    batch_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "processed_count": 0
    }
    
    # Process each file
    for file in files:
        if file.filename == '':
            continue
            
        # Save the uploaded file
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)
        
        # Check if it's a video
        is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv'))
        
        # Run inference
        result, error = run_inference(file_path, is_video)
        
        if not error:
            # Ensure we have metrics for each file
            if "precision" not in result:
                # Calculate metrics based on detections
                detections = result.get("detections", [])
                true_positives = sum(1 for d in detections if d["confidence"] > 0.7)
                false_positives = sum(1 for d in detections if d["confidence"] <= 0.7)
                false_negatives = 0  # We don't have ground truth
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                result["precision"] = precision
                result["recall"] = recall
                result["f1_score"] = f1_score
            
            # Update batch metrics
            batch_metrics["precision"] += result["precision"]
            batch_metrics["recall"] += result["recall"]
            batch_metrics["f1_score"] += result["f1_score"]
            batch_metrics["processed_count"] += 1
            
            # Update total detections and confidence
            detections = result.get("detections", [])
            total_detections += len(detections)
            total_confidence += sum(d["confidence"] for d in detections) if detections else 0
            
            results.append({
                "filename": file.filename,
                "output_path": result["output_path"],
                "detections": len(detections),
                "processing_time": result["processing_time"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"]
            })
        else:
            errors.append({"filename": file.filename, "error": error})
    
    # Calculate average metrics for this batch
    if batch_metrics["processed_count"] > 0:
        batch_metrics["precision"] /= batch_metrics["processed_count"]
        batch_metrics["recall"] /= batch_metrics["processed_count"]
        batch_metrics["f1_score"] /= batch_metrics["processed_count"]
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
        
        # Update global inference metrics with this batch's metrics
        inference_metrics["precision"] = batch_metrics["precision"]
        inference_metrics["recall"] = batch_metrics["recall"]
        inference_metrics["f1_score"] = batch_metrics["f1_score"]
        inference_metrics["total_detections"] = total_detections
        inference_metrics["files_processed"] = len(results)
        inference_metrics["avg_confidence"] = avg_confidence
    
    return jsonify({
        "results": results,
        "errors": errors,
        "total_detections": total_detections,
        "batch_metrics": {
            "precision": batch_metrics["precision"],
            "recall": batch_metrics["recall"],
            "f1_score": batch_metrics["f1_score"]
        }
    })

@app.route('/inference-results/<path:filename>')
def serve_result(filename):
    """Serve inference result files"""
    return send_from_directory('static/inference-results', filename)

@app.route('/templates/<path:filename>')
def serve_template(filename):
    """Serve template files"""
    return send_from_directory('templates', filename)

if __name__ == '__main__':
    # Create HTML template directory
    os.makedirs("templates", exist_ok=True)
    
    # Create index.html if it doesn't exist
    if not os.path.exists("templates/index.html"):
        with open("templates/index.html", "w") as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Federated Learning Inference Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #333;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .metric-card {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            flex: 1;
            min-width: 200px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .upload-container {
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .upload-container:hover {
            border-color: #aaa;
        }
        .results-container {
            margin-top: 20px;
        }
        .result-item {
            border-bottom: 1px solid #eee;
            padding: 10px 0;
        }
        .result-item:last-child {
            border-bottom: none;
        }
        .progress-bar {
            height: 10px;
            background-color: #e0e0e0;
            border-radius: 5px;
            margin-top: 5px;
        }
        .progress-bar-fill {
            height: 100%;
            background-color: #4CAF50;
            border-radius: 5px;
            transition: width 0.3s ease;
        }
        .hidden {
            display: none;
        }
        .loading {
            text-align: center;
            padding: 20px;
        }
        .video-container {
            margin-top: 10px;
        }
        video {
            max-width: 100%;
            max-height: 400px;
        }
        .batch-metrics {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f5e9;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>Federated Learning Inference Interface</h1>
    
    <div class="container">
        <h2>Model Information</h2>
        <div id="model-info">Loading model information...</div>
    </div>
    
    <div class="container">
        <h2>Run Inference</h2>
        <div class="upload-container" id="upload-area">
            <p>Drop files here or click to upload</p>
            <p><small>Supports images (.jpg, .jpeg, .png) and videos (.mp4, .avi, .mov)</small></p>
            <input type="file" id="file-input" multiple accept=".jpg,.jpeg,.png,.mp4,.avi,.mov" style="display: none;">
        </div>
        <button id="run-inference-btn" disabled>Run Inference</button>
        <div id="upload-status"></div>
    </div>
    
    <div class="container hidden" id="results-section">
        <h2>Inference Results</h2>
        <div class="batch-metrics" id="batch-metrics"></div>
        <div id="results-container"></div>
    </div>
    
    <div class="loading hidden" id="loading">
        <p>Running inference...</p>
    </div>

    <script>
        // DOM elements
        const modelInfoEl = document.getElementById('model-info');
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const runInferenceBtn = document.getElementById('run-inference-btn');
        const uploadStatus = document.getElementById('upload-status');
        const resultsSection = document.getElementById('results-section');
        const resultsContainer = document.getElementById('results-container');
        const loadingEl = document.getElementById('loading');
        const batchMetricsEl = document.getElementById('batch-metrics');
        
        // Selected files
        let selectedFiles = [];
        
        // Load model info when page loads
        window.addEventListener('DOMContentLoaded', loadModelInfo);
        
        // Set up file upload handlers
        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileSelect);
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#4CAF50';
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#ccc';
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#ccc';
            handleFileSelect({ target: { files: e.dataTransfer.files } });
        });
        
        // Run inference button
        runInferenceBtn.addEventListener('click', runInference);
        
        // Function to load model info
        async function loadModelInfo() {
            try {
                const response = await fetch('/model-info');
                const data = await response.json();
                
                // Create HTML for model info
                let html = `
                    <div class="metrics-container">
                        <div class="metric-card">
                            <h3>Model Name</h3>
                            <div>${data.model_name}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Model Type</h3>
                            <div>${data.model_type}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Status</h3>
                            <div>${data.model_status}</div>
                        </div>
                    </div>
                    
                    <h3>Federated Model Training Metrics</h3>
                    <div class="metrics-container">
                        <div class="metric-card">
                            <h3>Precision</h3>
                            <div class="metric-value">${(data.federated_metrics.precision * 100).toFixed(2)}%</div>
                        </div>
                        <div class="metric-card">
                            <h3>Recall</h3>
                            <div class="metric-value">${(data.federated_metrics.recall * 100).toFixed(2)}%</div>
                        </div>
                        <div class="metric-card">
                            <h3>mAP@50</h3>
                            <div class="metric-value">${(data.federated_metrics.mAP50 * 100).toFixed(2)}%</div>
                        </div>
                        <div class="metric-card">
                            <h3>mAP@50-95</h3>
                            <div class="metric-value">${(data.federated_metrics.mAP50_95 * 100).toFixed(2)}%</div>
                        </div>
                    </div>
                `;
                
                // Add dynamic inference metrics if available
                if (data.batch_metrics && data.batch_metrics.files_processed > 0) {
                    html += `
                        <h3>Dynamic Inference Metrics</h3>
                        <div class="metrics-container">
                            <div class="metric-card">
                                <h3>Precision</h3>
                                <div class="metric-value">${(data.batch_metrics.precision * 100).toFixed(2)}%</div>
                            </div>
                            <div class="metric-card">
                                <h3>Recall</h3>
                                <div class="metric-value">${(data.batch_metrics.recall * 100).toFixed(2)}%</div>
                            </div>
                            <div class="metric-card">
                                <h3>F1 Score</h3>
                                <div class="metric-value">${(data.batch_metrics.f1_score * 100).toFixed(2)}%</div>
                            </div>
                            <div class="metric-card">
                                <h3>Files Processed</h3>
                                <div class="metric-value">${data.batch_metrics.files_processed}</div>
                            </div>
                        </div>
                    `;
                }
                
                modelInfoEl.innerHTML = html;
            } catch (error) {
                modelInfoEl.innerHTML = `<p>Error loading model info: ${error.message}</p>`;
            }
        }
        
        // Function to handle file selection
        function handleFileSelect(event) {
            const files = event.target.files;
            if (!files.length) return;
            
            selectedFiles = Array.from(files);
            uploadStatus.innerHTML = `Selected ${selectedFiles.length} file(s)`;
            runInferenceBtn.disabled = false;
        }
        
        // Function to run inference
        async function runInference() {
            if (!selectedFiles.length) return;
            
            // Show loading
            loadingEl.classList.remove('hidden');
            resultsSection.classList.add('hidden');
            runInferenceBtn.disabled = true;
            
            // Create form data
            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            try {
                // Send request
                const response = await fetch('/run-inference', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                displayResults(data);
                
                // Refresh model info to show updated metrics
                loadModelInfo();
            } catch (error) {
                uploadStatus.innerHTML = `Error: ${error.message}`;
            } finally {
                loadingEl.classList.add('hidden');
                runInferenceBtn.disabled = false;
            }
        }
        
        // Function to display results
        function displayResults(data) {
            resultsSection.classList.remove('hidden');
            
            // Display batch metrics
            const batchMetrics = data.batch_metrics;
            batchMetricsEl.innerHTML = `
                <h3>Current Batch Results</h3>
                <div class="metrics-container">
                    <div class="metric-card">
                        <h3>Precision</h3>
                        <div class="metric-value">${(batchMetrics.precision * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Recall</h3>
                        <div class="metric-value">${(batchMetrics.recall * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>F1 Score</h3>
                        <div class="metric-value">${(batchMetrics.f1_score * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Total Detections</h3>
                        <div class="metric-value">${data.total_detections}</div>
                    </div>
                </div>
            `;
            
            // Display individual results
            let resultsHtml = '';
            
            data.results.forEach(result => {
                const isVideo = result.output_path.endsWith('.mp4') || 
                               result.output_path.endsWith('.avi') || 
                               result.output_path.endsWith('.mov');
                               
                resultsHtml += `
                    <div class="result-item">
                        <h3>${result.filename}</h3>
                        <p>Detections: ${result.detections} | Processing Time: ${result.processing_time.toFixed(2)}s</p>
                        <p>Precision: ${(result.precision * 100).toFixed(2)}% | Recall: ${(result.recall * 100).toFixed(2)}% | F1: ${(result.f1_score * 100).toFixed(2)}%</p>
                        ${isVideo ? 
                            `<div class="video-container">
                                <video controls>
                                    <source src="${result.output_path}" type="video/mp4">
                                    Your browser does not support the video tag.
                                </video>
                            </div>` : 
                            `<div class="image-container">
                                <img src="${result.output_path}" alt="Inference result" style="max-width: 100%; max-height: 400px;">
                            </div>`
                        }
                    </div>
                `;
            });
            
            // Display errors if any
            if (data.errors && data.errors.length) {
                resultsHtml += '<h3>Errors</h3>';
                data.errors.forEach(error => {
                    resultsHtml += `<div class="result-item error">
                        <p>${error.filename}: ${error.error}</p>
                    </div>`;
                });
            }
            
            resultsContainer.innerHTML = resultsHtml;
        }
    </script>
</body>
</html>
            """)
    
    # Load the model
    load_model()
    
    print("Starting Federated YOLOv5 App...")
    app.run(debug=True, port=5004)
