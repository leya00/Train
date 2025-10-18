# Train 🚉
A federated and machine learning–based application for real-time train detection and analysis. This project integrates traditional centralized machine learning with a federated learning approach to enable distributed training across multiple clients, ensuring data privacy, scalability, and efficiency.

# 📘 Overview

The Train Detection Dashboard uses computer vision and machine learning to identify and classify trains in image or video datasets. To enhance privacy and decentralization, this project implements Federated Learning (FL) — enabling multiple devices (clients) to collaboratively train a shared global model without exchanging raw data.

The system demonstrates how AI applications can be deployed securely and efficiently in transportation monitoring contexts, such as:

Train detection from surveillance footage

Safety monitoring on rail networks

Real-time analytics for transportation systems 

# 🧠 Machine Learning Component

The machine learning pipeline involves:

Data Preprocessing – Cleaning, normalization, and feature extraction from image/video inputs.

Model Architecture – A CNN-based detection model (e.g., YOLO or custom TensorFlow/PyTorch model) trained on train imagery datasets.

Training and Evaluation – Local training on client datasets, with accuracy, precision, and recall metrics.

Inference – Running the trained model on unseen data for detection and classification.

The ML model can operate independently in centralized environments or as part of the federated learning process. 

# 🌐 Federated Learning Component

The federated system enables distributed model training across multiple client nodes:

Client Nodes: Each client (e.g., local device, station server) holds its own dataset and trains a local model.

Central Server: The federated_app.py script coordinates model aggregation using the Federated Averaging (FedAvg) algorithm.

Privacy Preservation: Only model parameters (weights and gradients) are exchanged — not raw data — protecting sensitive visual data.

Communication Rounds: The system runs iterative communication rounds to synchronize global model updates until convergence. 

#⚙️ Installation & Setup
### 1️⃣ Prerequisites

Ensure the following are installed:

Python 3.8+

pip package manager

Virtual environment (recommended) 

### 2️⃣ Dependencies

Install required libraries:

`pip install -r requirements.txt` <br/>



(Ensure the requirements.txt includes flwr, torch or tensorflow, opencv-python, and numpy.)


(Ensure the requirements.txt includes flwr, torch or tensorflow, opencv-python, and numpy.)

## To run [In terminal/powershell]
move into the fl folder <br/>
`cd fl` <br/>
then run <br/>
`python federated_app.py`

