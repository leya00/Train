import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def train_model(model_path, labeled_dir, work_root, epochs=3, imgsz=640, nc=1, names="object"):
    """
    Train a YOLO model on a labeled dataset without requiring Flower server
    
    Args:
        model_path: Path to the base model to start from
        labeled_dir: Path to the labeled dataset (with images/ and labels/ subdirectories)
        work_root: Directory to store training outputs
        epochs: Number of training epochs
        imgsz: Image size for training
        nc: Number of classes
        names: Class names (comma-separated)
    
    Returns:
        Path to the best trained model
    """
    print(f"[STANDALONE] Starting training with model: {model_path}")
    print(f"[STANDALONE] Dataset: {labeled_dir}")
    print(f"[STANDALONE] Output directory: {work_root}")
    print(f"[STANDALONE] Training for {epochs} epochs")
    
    try:
        # Load the model
        model = YOLO(model_path)
        print(f"[STANDALONE] Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"[STANDALONE] Error loading model: {e}")
        print("[STANDALONE] Attempting to load default model...")
        model = YOLO('yolov8n.pt')
        print("[STANDALONE] Loaded default model")
    
    # Prepare directories
    train_folder = Path(work_root) / "train"
    val_folder = Path(work_root) / "val"
    out_root = Path(work_root) / "runs"
    
    for d in [train_folder, val_folder, out_root]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Prepare train/val split
    labeled_dir = Path(labeled_dir)
    img_dir = labeled_dir / "images"
    lbl_dir = labeled_dir / "labels"
    
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[STANDALONE] Error: Expected directory structure not found in {labeled_dir}")
        print(f"[STANDALONE] Looking for: {img_dir} and {lbl_dir}")
        return None
    
    # Get all images
    imgs = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not imgs:
        print(f"[STANDALONE] No images found in {img_dir}")
        return None
    
    # Split into train/val
    split = 0.8
    split_idx = max(1, int(len(imgs) * split)) if len(imgs) > 1 else 1
    sets = [("train", imgs[:split_idx]), ("val", imgs[split_idx:] or imgs[:1])]
    
    # Copy files to train/val directories
    for split_name, items in sets:
        dest_root = train_folder if split_name == "train" else val_folder
        (dest_root / "images").mkdir(parents=True, exist_ok=True)
        (dest_root / "labels").mkdir(parents=True, exist_ok=True)
        
        print(f"[STANDALONE] Copying {len(items)} images to {split_name} set")
        for img in items:
            lbl = lbl_dir / (img.stem + ".txt")
            # Use shutil.copy2 instead of os.system for cross-platform compatibility
            import shutil
            try:
                shutil.copy2(str(img), str(dest_root / "images" / img.name))
                if lbl.exists():
                    shutil.copy2(str(lbl), str(dest_root / "labels" / lbl.name))
            except Exception as e:
                print(f"[STANDALONE] Error copying files: {e}")
    
    # Create data.yaml
    data_yaml_path = out_root / "data.yaml"
    
    # Parse names
    class_names = names.split(",")
    
    # Write data.yaml content
    with open(data_yaml_path, "w") as f:
        f.write(f"train: {str(train_folder.resolve()).replace('\\', '/')}\n")
        f.write(f"val: {str(val_folder.resolve()).replace('\\', '/')}\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {class_names}\n")
    
    print(f"[STANDALONE] Created data configuration at {data_yaml_path}")
    
    # Train the model
    try:
        # Disable optimizer stripping to avoid issues
        os.environ['YOLO_STRIP_OPTIMIZER'] = 'false'
        
        # Start training
        print(f"[STANDALONE] Starting training for {epochs} epochs...")
        
        # Redirect stdout to capture training output
        import io
        import sys
        original_stdout = sys.stdout
        captured_output = io.StringIO()
        
        try:
            # Redirect stdout to capture output
            sys.stdout = captured_output
            
            # Run training
            model.train(
                data=str(data_yaml_path.resolve()),
                epochs=epochs,
                imgsz=imgsz,
                project=str(out_root),
                name="train_result",
                exist_ok=True,
                plots=True,
                save=True,
            )
            
            # Restore stdout
            sys.stdout = original_stdout
            print("[STANDALONE] Training completed successfully!")
            
            # Print captured output with proper encoding handling
            for line in captured_output.getvalue().splitlines():
                try:
                    print(line)
                except UnicodeEncodeError:
                    print(line.encode('ascii', 'replace').decode('ascii'))
                    
        except Exception as e:
            # Restore stdout in case of exception
            sys.stdout = original_stdout
            print(f"[STANDALONE] Error during training: {str(e)}")
            
            # Print any captured output
            try:
                for line in captured_output.getvalue().splitlines():
                    try:
                        print(line)
                    except UnicodeEncodeError:
                        print(line.encode('ascii', 'replace').decode('ascii'))
            except:
                pass
        
        # Find the best model
        best_model_path = out_root / "train_result" / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"[STANDALONE] Best model saved at: {best_model_path}")
            return str(best_model_path)
        else:
            # Try to find any model file
            model_files = list(out_root.glob("**/weights/*.pt"))
            if model_files:
                print(f"[STANDALONE] Found model at: {model_files[0]}")
                return str(model_files[0])
            else:
                print("[STANDALONE] No model file found after training")
                return None
        
    except Exception as e:
        print(f"[STANDALONE] Error during training: {e}")
        return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt", help="Base YOLO checkpoint")
    p.add_argument("--labeled_dir", required=True, help="Path to labeled dataset (images/ & labels/)")
    p.add_argument("--work_root", required=True, help="Working directory for outputs")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--names", default="object")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trained_model = train_model(
        model_path=args.model,
        labeled_dir=args.labeled_dir,
        work_root=args.work_root,
        epochs=args.epochs,
        imgsz=args.imgsz,
        nc=args.nc,
        names=args.names
    )
    
    if trained_model:
        print(f"[STANDALONE] Training successful. Model saved at: {trained_model}")
        sys.exit(0)
    else:
        print("[STANDALONE] Training failed.")
        sys.exit(1)
