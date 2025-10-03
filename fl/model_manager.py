import os
import requests
import hashlib
import json
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import torch
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages model loading from GitHub and local sources with caching and version checking
    """
    
    def __init__(self, cache_dir="model_cache", config_file="model_config.json", models_dir="models"):
        self.cache_dir = Path(cache_dir)
        self.config_file = Path(config_file)
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / "registry.json"
        
        # Create necessary directories
        self.cache_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # GitHub repository configuration
        self.github_repo = "KaitlynKK/fed-learning7-belalandkaitlyn"
        self.github_branch = "main"
        self.github_models_path = "static/output"
        
        # Model configuration
        self.config = self._load_config()
        self.current_model = None
        self.model_info = {}
        
    def _load_config(self):
        """Load model configuration from file"""
        default_config = {
            "model_source": "github",  # "github", "local", "custom"
            "github_model_name": "final_model.pt",
            "local_model_path": "models/local_model.pt",
            "custom_model_url": "",
            "cache_duration_hours": 24,
            "auto_update": True,
            "fallback_to_default": True,
            "last_update_check": None,
            "model_versions": {}
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        
        return default_config
    
    def _save_config(self):
        """Save model configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")
    
    def _get_github_headers(self):
        """Get headers for GitHub API requests"""
        headers = {"Accept": "application/vnd.github.v3+json"}
        return headers
    
    def _get_github_file_url(self, filename):
        """Get the raw GitHub URL for a file"""
        return f"https://raw.githubusercontent.com/{self.github_repo}/{self.github_branch}/{self.github_models_path}/{filename}"
    
    def _get_github_api_url(self, filename=None):
        """Get GitHub API URL for file information or directory listing"""
        if filename:
            return f"https://api.github.com/repos/{self.github_repo}/contents/{self.github_models_path}/{filename}"
        else:
            return f"https://api.github.com/repos/{self.github_repo}/contents/{self.github_models_path}"
    
    def _get_file_hash(self, filepath):
        """Get MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to get hash for {filepath}: {e}")
            return None
    
    def _check_github_model_info(self, filename):
        """Check if GitHub model has been updated using GitHub API"""
        try:
            api_url = self._get_github_api_url(filename)
            headers = self._get_github_headers()
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            file_info = response.json()
            return {
                "sha": file_info.get("sha"),
                "size": file_info.get("size"),
                "download_url": file_info.get("download_url"),
                "updated_at": file_info.get("updated_at")
            }
        except Exception as e:
            logger.error(f"Failed to check GitHub model info: {e}")
            return None
    
    def _download_github_model(self, filename, force_download=False):
        """Download model from GitHub with caching and version checking"""
        cache_path = self.cache_dir / filename
        
        # Check if we need to update
        should_download = force_download
        
        if not should_download and cache_path.exists():
            # Check if cache is still valid
            last_check = self.config.get("last_update_check")
            if last_check:
                last_check_time = datetime.fromisoformat(last_check)
                cache_duration = timedelta(hours=self.config.get("cache_duration_hours", 24))
                if datetime.now() - last_check_time < cache_duration:
                    logger.info(f"Using cached model: {cache_path}")
                    return str(cache_path)
            
            # Check GitHub for updates
            if self.config.get("auto_update", True):
                github_info = self._check_github_model_info(filename)
                if github_info:
                    cached_hash = self._get_file_hash(cache_path)
                    if cached_hash != github_info["sha"][:32]:  # GitHub SHA is longer
                        logger.info(f"Model updated on GitHub, downloading new version")
                        should_download = True
        
        if should_download or not cache_path.exists():
            try:
                # Download the model from public repository
                download_url = self._get_github_file_url(filename)
                logger.info(f"Downloading model from GitHub: {download_url}")
                
                response = requests.get(download_url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Save to cache
                with open(cache_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Update config with new hash and timestamp
                file_hash = self._get_file_hash(cache_path)
                if file_hash:
                    self.config["model_versions"][filename] = {
                        "hash": file_hash,
                        "downloaded_at": datetime.now().isoformat(),
                        "size": cache_path.stat().st_size
                    }
                self.config["last_update_check"] = datetime.now().isoformat()
                self._save_config()
                
                logger.info(f"Successfully downloaded model to: {cache_path}")
                
            except Exception as e:
                logger.error(f"Failed to download model from GitHub: {e}")
                if cache_path.exists():
                    logger.info(f"Using existing cached model: {cache_path}")
                    return str(cache_path)
                else:
                    return None
        
        return str(cache_path) if cache_path.exists() else None
    
    def _load_yolo_model(self, model_path):
        """Load YOLO model with error handling for different formats"""
        try:
            # Try Ultralytics YOLO first (newer format)
            model = YOLO(model_path)
            logger.info(f"Successfully loaded model with Ultralytics YOLO: {model_path}")
            return model, "ultralytics"
        except Exception as e:
            logger.warning(f"Ultralytics YOLO failed: {e}")
            
            try:
                # Try torch.hub with error handling
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
                logger.info(f"Successfully loaded model with torch.hub: {model_path}")
                return model, "torch_hub"
            except Exception as e2:
                if "'Detect' object has no attribute 'grid'" in str(e2):
                    # Fix the grid attribute issue
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        
                        # Fix the checkpoint before loading
                        if 'model' in checkpoint:
                            model_dict = checkpoint['model']
                            for key, value in model_dict.items():
                                if hasattr(value, '__class__') and 'Detect' in str(value.__class__):
                                    if not hasattr(value, 'grid'):
                                        value.grid = [torch.zeros(1)] * 3
                                    if not hasattr(value, 'anchor_grid'):
                                        value.anchor_grid = [torch.zeros(1)] * 3
                        
                        # Save the fixed checkpoint temporarily
                        temp_path = model_path + '_fixed.pt'
                        torch.save(checkpoint, temp_path)
                        
                        # Load the fixed model
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path=temp_path, force_reload=True)
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        logger.info(f"Successfully loaded fixed model: {model_path}")
                        return model, "torch_hub_fixed"
                    except Exception as e3:
                        logger.error(f"Failed to fix and load model: {e3}")
                else:
                    logger.error(f"Failed to load model with torch.hub: {e2}")
                
                return None, None
    
    def load_model(self, source=None, model_path=None, force_download=False):
        """
        Load model from specified source
        
        Args:
            source: "github", "local", "custom", or None (use config)
            model_path: Path to local model or custom URL
            force_download: Force download from GitHub even if cached
        """
        if source is None:
            source = self.config.get("model_source", "github")
        
        model_path = None
        model_source_info = {}
        
        if source == "github":
            filename = self.config.get("github_model_name", "final_model.pt")
            model_path = self._download_github_model(filename, force_download)
            model_source_info = {
                "source": "github",
                "filename": filename,
                "cached_path": model_path,
                "repository": self.github_repo
            }
            
        elif source == "local":
            if model_path is None:
                model_path = self.config.get("local_model_path", "models/local_model.pt")
            model_source_info = {
                "source": "local",
                "path": model_path
            }
            
        elif source == "custom":
            if model_path is None:
                model_path = self.config.get("custom_model_url", "")
            if model_path.startswith("http"):
                # Download custom model
                try:
                    response = requests.get(model_path, timeout=30)
                    response.raise_for_status()
                    
                    filename = os.path.basename(model_path)
                    cache_path = self.cache_dir / filename
                    
                    with open(cache_path, 'wb') as f:
                        f.write(response.content)
                    
                    model_path = str(cache_path)
                    logger.info(f"Downloaded custom model to: {cache_path}")
                except Exception as e:
                    logger.error(f"Failed to download custom model: {e}")
                    return None, None
            
            model_source_info = {
                "source": "custom",
                "original_path": model_path
            }
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            
            # Try fallback to default YOLOv5 model
            if self.config.get("fallback_to_default", True):
                logger.info("Attempting to load default YOLOv5 model as fallback...")
                try:
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                    logger.info("Successfully loaded default YOLOv5 model")
                    return model, "default_yolov5"
                except Exception as e:
                    logger.error(f"Failed to load default YOLOv5 model: {e}")
            
            return None, None
        
        # Load the model
        model, model_type = self._load_yolo_model(model_path)
        
        if model is not None:
            self.current_model = model
            self.model_info = {
                **model_source_info,
                "model_type": model_type,
                "model_path": model_path,
                "loaded_at": datetime.now().isoformat()
            }
            logger.info(f"Model loaded successfully: {self.model_info}")
        
        return model, model_type
    
    def get_model_info(self):
        """Get information about the currently loaded model"""
        return self.model_info
    
    def update_model_source(self, source, **kwargs):
        """Update the model source configuration"""
        self.config["model_source"] = source
        
        if source == "github":
            if "model_name" in kwargs:
                self.config["github_model_name"] = kwargs["model_name"]
        elif source == "local":
            if "model_path" in kwargs:
                self.config["local_model_path"] = kwargs["model_path"]
        elif source == "custom":
            if "model_url" in kwargs:
                self.config["custom_model_url"] = kwargs["model_url"]
        
        self._save_config()
        logger.info(f"Updated model source to: {source}")
    
    def list_available_models(self):
        """List available models from GitHub"""
        try:
            api_url = self._get_github_api_url()
            headers = self._get_github_headers()
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            files = response.json()
            models = []
            for file in files:
                if file["name"].endswith(('.pt', '.pth', '.onnx')):
                    models.append({
                        "name": file["name"],
                        "size": file["size"],
                        "download_url": file["download_url"],
                        "updated_at": file["updated_at"]
                    })
            
            return models
        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            return []
    
    def clear_cache(self):
        """Clear the model cache"""
        try:
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.info("Model cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_info(self):
        """Get information about cached models"""
        cache_info = {
            "cache_dir": str(self.cache_dir),
            "cached_models": [],
            "total_size": 0
        }
        
        for file in self.cache_dir.glob("*"):
            if file.is_file():
                stat = file.stat()
                cache_info["cached_models"].append({
                    "name": file.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
                cache_info["total_size"] += stat.st_size
        
        return cache_info
        
    def _load_model_registry(self):
        """Load the model registry from file"""
        if not self.registry_file.exists():
            return {
                "models": [
                    {
                        "name": "default_model",
                        "file": "default_model.pt",
                        "created_at": datetime.now().isoformat()
                    }
                ],
                "last_updated": datetime.now().isoformat()
            }
        
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            return {
                "models": [],
                "last_updated": datetime.now().isoformat()
            }
    
    def _save_model_registry(self, registry):
        """Save the model registry to file"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def save_model(self, model, model_name):
        """
        Save a model with a custom name
        
        Args:
            model: The model object to save
            model_name: The name to give the model
        
        Returns:
            dict: Information about the saved model
        """
        # Sanitize model name for filename
        safe_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
        if not safe_name:
            safe_name = "unnamed_model"
        
        # Create filename with .pt extension
        if not safe_name.endswith('.pt'):
            filename = f"{safe_name}.pt"
        else:
            filename = safe_name
        
        # Full path to save the model
        model_path = self.models_dir / filename
        
        try:
            # Save the model
            model.save(model_path)
            logger.info(f"Model saved successfully to {model_path}")
            
            # Update the registry
            registry = self._load_model_registry()
            
            # Check if model with this name already exists
            existing_model = next((m for m in registry["models"] if m["name"] == model_name), None)
            
            if existing_model:
                # Update existing entry
                existing_model["file"] = filename
                existing_model["updated_at"] = datetime.now().isoformat()
            else:
                # Add new entry
                registry["models"].append({
                    "name": model_name,
                    "file": filename,
                    "created_at": datetime.now().isoformat()
                })
            
            registry["last_updated"] = datetime.now().isoformat()
            self._save_model_registry(registry)
            
            return {
                "name": model_name,
                "file": filename,
                "path": str(model_path),
                "saved_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            return None
    
    def list_saved_models(self):
        """
        List all saved models from the registry
        
        Returns:
            list: List of saved models
        """
        registry = self._load_model_registry()
        return registry["models"]
    
    def load_saved_model(self, model_name=None, model_file=None):
        """
        Load a saved model by name or filename
        
        Args:
            model_name: Name of the model to load
            model_file: Filename of the model to load
            
        Returns:
            tuple: (model, model_type)
        """
        registry = self._load_model_registry()
        
        # Find the model in the registry
        if model_name:
            model_entry = next((m for m in registry["models"] if m["name"] == model_name), None)
        elif model_file:
            model_entry = next((m for m in registry["models"] if m["file"] == model_file), None)
        else:
            # Default to the first model in the registry
            model_entry = registry["models"][0] if registry["models"] else None
        
        if not model_entry:
            logger.error(f"Model not found: {model_name or model_file}")
            return None, None
        
        # Full path to the model file
        model_path = str(self.models_dir / model_entry["file"])
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, None
        
        # Load the model
        model, model_type = self._load_yolo_model(model_path)
        
        if model is not None:
            self.current_model = model
            self.model_info = {
                "source": "saved",
                "name": model_entry["name"],
                "file": model_entry["file"],
                "model_type": model_type,
                "model_path": model_path,
                "loaded_at": datetime.now().isoformat()
            }
            logger.info(f"Saved model loaded successfully: {model_entry['name']}")
        
        return model, model_type
