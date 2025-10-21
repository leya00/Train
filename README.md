# 🚆 Train Detection Dashboard

A privacy-preserving **Machine Learning (ML)** and **Federated Learning (FL)** system for **train detection and analytics**.  
This project demonstrates how distributed AI can support transportation monitoring, railway safety, and real-time analytics — all while keeping sensitive video data secure on the edge.

---

## 🧭 Table of Contents
- [Key Features](#key-features)
- [Architecture](#architecture)
- [System-Demonstration](#system-demonstration)
    - [1. Purpose](#1️.Purpose)
    - [2. System Structure](#2.System-Structure)
    - [3. Components](#3.Components)
    - [4. Workflow](#4.Workflow)
    - [5. API Overview](#5.API-Overview)
    - [6. Typical Use Scenarios](#6.Typical-Use-Scenarios)
    - [7. Security & Privacy](#7.Security-&-Privacy)
    - [8. Future Improvements](#8.Future-Improvements)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [1) Centralised ML Demo](#1-centralised-ml-demo)
  - [2) Federated Learning (Server + Clients)](#2-federated-learning-server--clients)
  - [3) Dashboard / App](#3-dashboard--app)
- [Data & Model Notes](#data--model-notes)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## 🚀 Key Features

- **Train Detection** using a YOLOv5-based deep learning model.
- **Federated Learning (FL)** implementation allowing mulitple clients to train locally while sharing only model weights.
- ** Privacy-perserving training** ], ensuring raw data remains on local clients.
- **Dashboard integration** for uploading videos or images and visualising predictions.
- **Lightweight, modular design** for rapid documentation and scaling.

---

## 🏗️ Architecture  

```text
                +-------------------+
                |   Aggregation     |
                |     Server        |
                | (FedAvg, logging) |
                +---------+---------+
                          ^
                          | model updates
             -------------+-------------
             |                           |
+-------------------+         +-------------------+
|   Client A        |         |   Client B        |
| Local data        |         | Local data        |
| Train locally     |         | Train locally     |
| send weights -->  |         | send weights -->  |
+-------------------+         +-------------------+
```

- **Server ('server.py')** - Coordinate communication rounds and aggregates local model updates.
- **Clients ('client.py)'** - Train models on their local datasets and send only weights/gradients back to the server.
- **Aggregator** – Uses the **Federated Averaging (FedAvg)** algorithm to merge model updates into a unified global model.

---  

## System Demonstration (End-to-End) 
#### 1️. Purpose
   
Showcasing how machine learning and federated learning can be applied for privacy aware object detection across distributed sites. It can solve problems related to data centralization, privacy, and collaboration between mulitple nodes.

Who uses it? 
- Researches & Students learning Federated Learning (FL)
- Developers integrating YOLO-based models on model based detection
- Project evaluators and stakeholders viewing a working demo

#### 2️. System Structure
User Interface (Flask)
    │
    ├── API Requests (HTTP)
    │
    ├── YOLOv5 Inference Engine
    │
    └── Federated Learning (Flower)
          ├── FL Server (Coordinator)
          └── FL Clients (Local Nodes)
Data never leaves the client, only model weights are shared.
#### 3️. Components
| Component            | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| **Frontend/UI**      | Flask-based web dashboard to trigger inference and training |
| **Backend**          | Flask API routes handling training and inference logic      |
| **YOLOv5 Model**     | Performs object detection on train images/videos            |
| **Flower FL Server** | Coordinates training rounds and aggregates models           |
| **FL Clients**       | Each client trains locally on private datasets              |
| **Data Storage**     | Stores training results, model weights, and logs            |

  #### 4. Workflow
  a. User uploads data or triggers an inference from the UI
  b. Flask sends request -> YOLOv5 runs detection -> results displayed
  c. For FL mode:
      i. FL server starts (*federated_app.py*)
     ii. Clients train locally.
     iii. Updates sent to server 
  d. Global model improves collaboratively without data sharing.
  
#### 5. API Overview
| Endpoint                 | Method | Description                   |
| ------------------------ | ------ | ----------------------------- |
| `/inference`             | POST   | Run YOLOv5 inference          |
| `/train/central`         | POST   | Start centralized training    |
| `/train/federated/start` | POST   | Start federated training      |
| `/status`                | GET    | Check job and training status |
| `/artifacts`             | GET    | Access model outputs and logs |


   
#### 6️. Typical Uses Systems
✅ Inference Mode

Upload an image or video → Run YOLOv5 → See detection results.

🔁 Federated Mode

Start Flower server → Connect local clients → Train collaboratively → Aggregate global model.
    
#### 7. Security & Privacy
- Federated setup ensures data stays local
- Central server receives only model paramters, not imgaes
- Environement variables handles secrets
- Logs avoid storing sensitive details

#### 8. Future improvements
- Integrate latest version of YOLO/RT-DETR models (GroundingDino if it needs a larger scale of model-based detection)
- Add a lively dashboard for FL progress
 

## 🗂 Repository Structure 
.
├─ backend/                 # Backend service code (API endpoints, dashboard logic)
├─ fl/                      # Federated learning utilities and scripts
├─ data/                    # Dataset directory (local client data)
├─ frames/video_1/          # Example extracted video frames
├─ model/train/             # Model checkpoints and logs
├─ static/uploads/          # Directory for uploaded files
├─ client_yolo.py           # FL client (YOLOv5)
├─ server_yolo.py           # FL server (FedAvg)
├─ federated_yolov5_app.py  # Federated learning app entry point
├─ requirements.txt         # Python dependencies
└─ .gitmodules              # Submodules (e.g., YOLOv5 detector) 

--- 

## ⚙️ Prerequisites 
- Python +3.9 (Python 3.10 recommended)
- **pip** and **venv**
- (Optional) NVIDIA GPU with CUDA support for faster training
- **ffmpeg** (for video-to-frame extraction)**
Install dependencies:
```
pip install -r requirements.txt
```

--- 

## 🧠 Quick Start 
#### 1️⃣ Centralised ML Demo 
If you want to run a quick local inference: 
```
# 1. Clone Repository
git clone https://github.com/leya00/Train.git
cd Train

# 2. Set up environment
python -m -venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run centralised detection demo
python federated_yolov5_app.py
```
Ensure your input frames or videos are stored in:

- frames/video_1/
- data/
- static/uploads/

#### 2️⃣ Federated Learning (Server + Clients)
##### 🖥️ Start the Server
```python server_yolo.py --rounds 5 --port 8080```

💻 Start Client(s)
##### Example for Client A
```python client_yolo.py --server http://127.0.0.1:8080 --data ./data/client_a --epochs 1 --batch-size 8```

##### Example for Client B
```python client_yolo.py --server http://127.0.0.1:8080 --data ./data/client_b --epochs 1 --batch-size 8```

##### 🧩 Single-Process FL Demo
```python federated_yolov5_app.py``` 

#### 3️⃣ Dashboard / App

A backend dashboard is available for uploading videos and visualising detections.

To run:
```
cd backend
export FLASK_APP=app.py
flask run 
```
Access it via your browser at http://127.0.0.1:5000 

## 📦 Data & Model Notes

- Sample Frames: Provided under frames/video_1/.
- Datasets: Add local training data to data/client_a/, data/client_b/, etc.
- Checkpoints: Saved in model/train/ after training.

Ensure each client dataset follows YOLOv5 directory format (images and labels). 

## ⚙️ Configuration

Common arguments and flags: 
| Parameter      | Description                          |
| -------------- | ------------------------------------ |
| `--epochs`     | Number of training epochs per client |
| `--batch-size` | Training batch size                  |
| `--img-size`   | Input image resolution               |
| `--server`     | Server address for FL communication  |
| `--rounds`     | Number of federated rounds           |
| `--data`       | Dataset directory path               |
| `--weights`    | Path to pretrained weights           |

## 🛠️ Troubleshooting 
| Issue                  | Solution                                                         |
| ---------------------- | ---------------------------------------------------------------- |
| **Submodule missing**  | Run `git submodule update --init --recursive`                    |
| **CUDA not available** | Ensure GPU drivers and CUDA toolkit are installed                |
| **Image path errors**  | Verify paths in `frames/video_1/`, `data/`, or `static/uploads/` |
| **Port in use**        | Change port using `--port` flag                                  |
| **No detections**      | Check model weights and dataset labels                           |

## 🧭 Roadmap 
- Integrate real-time video streaming
- Add evaluation metrics dashboard (Precision, Recall, F1)
- Support YOLOv8-based training
- Add Docker container deployment
- Improve frontend visualisation with Plotly/Chart.js

## 🤝 Contributing
Contributions are welcome!
Please: 
- Fork this repository
- Create a feature branch
- Commit changes with descriptive messages
- Open a Pull Request (PR)

## 🙏 Acknowledgements

- [**YOLOv5**](https://github.com/ultralytics/yolov5) — for the base detection framework  
- [**Flower**](https://flower.dev) — for federated learning inspiration  
- [**OpenCV**](https://opencv.org) — for image processing  
- [**Flask**](https://flask.palletsprojects.com) — for backend services
- Optional [**GroundingDino**](https://github.com/IDEA-Research/GroundingDINO) - this was another model for machine learning but due to heavy training requirements, it was close to optimised.

## 🧩 Author

Developed by:
Joseph Linao, Belal Nur, Andrew Yang, Shania Chu Pui Chan, Kaitlyn Chan, Leya Asmeron — integrating Machine Learning, Federated Learning, and Web Deployment for AI-driven transport analytics.
