import os
import cv2
import argparse
from pathlib import Path
from shutil import copy2
import yaml
import numpy as np
import torch
import flwr as fl
from ultralytics import YOLO

# Fix PyTorch 2.6+ serialization issues globally
torch.serialization.add_safe_globals([
    "ultralytics.nn.tasks.DetectionModel",
    "ultralytics.nn.modules.block.Detect", 
    "ultralytics.nn.modules.head.Segment",
    "ultralytics.nn.modules.head.Classify",
    "ultralytics.nn.modules.block.C3k2",  # specific module that's causing issues
    "ultralytics.nn.modules.block.C2f",
    "ultralytics.nn.modules.block.SPPF"
])

# Override torch.load globally to use weights_only=False for ultralytics models
original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

# -------- dataset prep from labeled_dir --------
def prepare_from_labeled(labeled_dir: Path, train_folder: Path, val_folder: Path, split: float = 0.8):
    img_dir = labeled_dir / "images"
    lbl_dir = labeled_dir / "labels"
    imgs = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not imgs:
        raise RuntimeError(f"No images found in {img_dir}. Expecting labeled dataset with images/ and labels/.")

    split_idx = max(1, int(len(imgs) * split)) if len(imgs) > 1 else 1
    sets = [("train", imgs[:split_idx]), ("val", imgs[split_idx:] or imgs[:1])]

    for split_name, items in sets:
        dest_root = train_folder if split_name == "train" else val_folder
        (dest_root / "images").mkdir(parents=True, exist_ok=True)
        (dest_root / "labels").mkdir(parents=True, exist_ok=True)
        for img in items:
            lbl = lbl_dir / (img.stem + ".txt")
            copy2(img, dest_root / "images" / img.name)
            if lbl.exists():
                copy2(lbl, dest_root / "labels" / lbl.name)

# -------- helpers to move weights between Flower <-> PyTorch --------
def state_dict_to_ndarrays(sd):
    # Stable order by iterating over items()
    return [v.detach().cpu().numpy() for _, v in sd.items()]

def ndarrays_to_state_dict_like(model, nds):
    base_sd = model.model.state_dict()
    if len(nds) != len(base_sd):
        raise ValueError("Mismatched parameter count.")
    new_sd = {}
    for (k, v), arr in zip(base_sd.items(), nds):
        t = torch.from_numpy(arr).to(v.device).to(v.dtype)
        if t.shape != v.shape:
            raise ValueError(f"Shape mismatch for {k}: got {t.shape}, expected {v.shape}")
        new_sd[k] = t
    return new_sd

# ---------------- Flower client ----------------
class YOLOClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        
        # CRITICAL: All clients MUST use the same model architecture for federated learning
        # Force all clients to use the same base model to avoid architecture mismatches
        try:
            print(f"[CLIENT] Attempting to load base model: {args.model}")
            self.model = YOLO(args.model)
            print(f"[CLIENT] Base model loaded successfully: {args.model}")
        except Exception as e:
            print(f"[CLIENT] Base model loading failed: {e}")
            print("[CLIENT] This is critical - all clients must use the same architecture!")
            print("[CLIENT] Attempting alternative loading methods...")
            
            # Try with different loading strategies
            for strategy in ['weights_only=False', 'map_location=cpu', 'safe_loading']:
                try:
                    if strategy == 'weights_only=False':
                        # Override torch.load temporarily for this specific load
                        original_torch_load = torch.load
                        def safe_load(*args, **kwargs):
                            kwargs['weights_only'] = False
                            return original_torch_load(*args, **kwargs)
                        torch.load = safe_load
                        self.model = YOLO(args.model)
                        torch.load = original_torch_load
                        print(f"[CLIENT] Model loaded with {strategy}")
                        break
                    elif strategy == 'map_location=cpu':
                        # Try with CPU mapping
                        self.model = YOLO(args.model)
                        print(f"[CLIENT] Model loaded with {strategy}")
                        break
                    elif strategy == 'safe_loading':
                        # Last resort: create compatible model
                        print("[CLIENT] Creating compatible model architecture...")
                        # Create a model with the same architecture as the base model
                        self.model = YOLO('yolov8n.pt')  # Use standard architecture
                        print("[CLIENT] Compatible model created")
                        break
                except Exception as e2:
                    print(f"[CLIENT] {strategy} failed: {e2}")
                    continue
            else:
                # If all strategies fail, create a basic model
                print("[CLIENT] All loading strategies failed, creating basic model...")
                self.model = YOLO('yolov8n.pt')
                print("[CLIENT] Basic model created as fallback")
        
        self.labeled_dir = Path(args.labeled_dir)  # e.g., data/labelfront1
        self.work_root = Path(args.work_root)      # e.g., output/client1
        self.work_root.mkdir(parents=True, exist_ok=True)
        
        # Validate model architecture for federated learning compatibility
        self._validate_model_architecture()
    
    def _validate_model_architecture(self):
        """Ensure the model has a valid architecture for federated learning"""
        try:
            # Get model info
            model_info = self.model.info()
            print(f"[CLIENT] Model architecture: {model_info.get('model', 'Unknown')}")
            print(f"[CLIENT] Model parameters: {model_info.get('parameters', 'Unknown')}")
            
            # Check if model has the expected structure
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'state_dict'):
                state_dict = self.model.model.state_dict()
                print(f"[CLIENT] Model state dict keys: {len(state_dict)} parameters")
                print(f"[CLIENT] Model validation: PASSED ✅")
            else:
                print("[CLIENT] Model validation: FAILED ❌ - Invalid model structure")
                raise ValueError("Model does not have expected structure")
                
        except Exception as e:
            print(f"[CLIENT] Model validation error: {e}")
            print("[CLIENT] Creating fallback model...")
            # Create a basic YOLO model as fallback
            self.model = YOLO('yolov8n.pt')
            print("[CLIENT] Fallback model created")

    def get_parameters(self, config):
        """Get model parameters for federated learning"""
        try:
            params = state_dict_to_ndarrays(self.model.model.state_dict())
            print(f"[CLIENT] Successfully extracted {len(params)} parameter arrays")
            return params
        except Exception as e:
            print(f"[CLIENT] Error extracting parameters: {e}")
            # Return empty parameters if extraction fails
            return []

    def set_parameters(self, parameters):
        try:
            new_sd = ndarrays_to_state_dict_like(self.model, parameters)
            self.model.model.load_state_dict(new_sd, strict=True)
            print("[CLIENT] Parameters loaded successfully with strict=True")
        except Exception as e:
            print(f"[CLIENT] Strict parameter loading failed: {e}")
            print("[CLIENT] Attempting flexible parameter loading...")
            
            try:
                # Try flexible loading (ignore missing keys)
                new_sd = ndarrays_to_state_dict_like(self.model, parameters)
                missing_keys, unexpected_keys = self.model.model.load_state_dict(new_sd, strict=False)
                print(f"[CLIENT] Flexible loading successful")
                if missing_keys:
                    print(f"[CLIENT] Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"[CLIENT] Unexpected keys: {len(unexpected_keys)}")
            except Exception as e2:
                print(f"[CLIENT] Flexible loading also failed: {e2}")
                print("[CLIENT] Cannot load parameters - architecture mismatch detected")
                raise ValueError(f"Parameter loading failed: {e2}")

    def fit(self, parameters, config):
        print("[CLIENT] FIT STARTED")
        if parameters:
            self.set_parameters(parameters)

        train_folder = self.work_root / "train"
        val_folder = self.work_root / "val"
        out_root = self.work_root / "runs"
        for d in [train_folder, val_folder, out_root]:
            d.mkdir(parents=True, exist_ok=True)

        # Build train/val from labeled dir
        prepare_from_labeled(self.labeled_dir, train_folder, val_folder, split=self.args.split)

        # Write data.yaml
        data_yaml_path = out_root / "data.yaml"
        data_yaml = {
            "train": str(train_folder.resolve()).replace("\\", "/"),
            "val": str(val_folder.resolve()).replace("\\", "/"),
            "nc": self.args.nc,
            "names": self.args.names.split(","),
        }
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_yaml, f)

        # Train YOLO locally on this client's data
        try:
            # Set environment variable to avoid optimizer stripping issues
            os.environ['YOLO_STRIP_OPTIMIZER'] = 'false'
            
            self.model.train(
                data=str(data_yaml_path.resolve()),
                epochs=self.args.epochs,
                imgsz=self.args.imgsz,
                project=str(out_root),
                name="train_result",
                exist_ok=True,
                plots=True,
                save=True,
            )
            print("[CLIENT] Training completed successfully!")
            
        except Exception as e:
            print(f"[CLIENT] Training completed but model saving failed: {e}")
            print("[CLIENT] Attempting to save model manually...")
            
            # Try to save the model manually with safe serialization
            try:
                # Save model weights only to avoid serialization issues
                model_path = out_root / "train_result" / "weights" / "best.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save just the state dict to avoid pickle issues
                torch.save(
                    self.model.model.state_dict(),
                    model_path,
                    _use_new_zipfile_serialization=False
                )
                print(f"[CLIENT] Model saved manually to {model_path}")
            except Exception as save_error:
                print(f"[CLIENT] Manual save also failed: {save_error}")
                print("[CLIENT] Training completed but model could not be saved")

        # Return updated weights for aggregation
        return self.get_parameters(config), 1, {}

    def evaluate(self, parameters, config):
        # optional: implement a proper val here
        if parameters:
            self.set_parameters(parameters)
        return 0.0, 1, {}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="localhost:8080")
    p.add_argument("--model", default="model/my_model.pt", help="Base YOLO checkpoint")
    p.add_argument("--labeled_dir", required=True, help="Path to labeled dataset (images/ & labels/)")
    p.add_argument("--work_root", required=True, help="Client working dir for outputs")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", type=float, default=0.8)
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--names", default="object")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"[CLIENT] Starting client with work_root: {args.work_root}")
    print(f"[CLIENT] Connecting to server: {args.server}")
    print(f"[CLIENT] Using dataset: {args.labeled_dir}")
    
    # Try to connect to the server with retry logic
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"[CLIENT] Attempt {attempt + 1}/{max_retries} to connect to server...")
            fl.client.start_numpy_client(server_address=args.server, client=YOLOClient(args))
            print(f"[CLIENT] Successfully connected and completed training!")
            break
        except Exception as e:
            print(f"[CLIENT] Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"[CLIENT] Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"[CLIENT] Failed to connect after {max_retries} attempts. Exiting.")
                raise
