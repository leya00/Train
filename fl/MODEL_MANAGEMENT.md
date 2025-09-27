# Model Management System

This system provides automatic model pulling from GitHub, caching, version checking, and support for local user-trained models.

## Features

- **Automatic GitHub Model Pulling**: Automatically downloads the latest trained model from your public GitHub repository
- **Smart Caching**: Caches models locally to avoid repeated downloads
- **Version Checking**: Checks for model updates using GitHub API
- **Local Model Support**: Support for user-trained models on local devices
- **Multiple Model Sources**: GitHub, local files, or custom URLs
- **Configuration Management**: Persistent configuration with JSON settings
- **CLI Interface**: Command-line tool for model management
- **REST API**: Web API endpoints for model management

## Quick Start

### 1. Automatic Model Loading (Default)

The system automatically pulls the latest model from GitHub on startup:

```python
from model_manager import ModelManager

# Initialize model manager
model_manager = ModelManager()

# Load model (automatically uses GitHub by default)
model, model_type = model_manager.load_model()
```

### 2. Using the Federated App

The federated app now automatically uses the ModelManager:

```bash
cd fl
python federated_app.py
```

The app will automatically:
- Download the latest model from GitHub
- Cache it locally
- Check for updates periodically
- Fall back to default YOLOv5 if needed

## Model Sources

### GitHub (Default)
- **Repository**: `KaitlynKK/fed-learning7-belalandkaitlyn` (public)
- **Path**: `static/output/`
- **Default Model**: `final_model.pt`
- **Auto-update**: Enabled by default

### Local Models
For user-trained models on local devices:

```python
# Load local model
model, model_type = model_manager.load_model(
    source="local", 
    model_path="path/to/your/model.pt"
)
```

### Custom URLs
For models hosted elsewhere:

```python
# Load from custom URL
model, model_type = model_manager.load_model(
    source="custom", 
    model_path="https://example.com/model.pt"
)
```

## Configuration

### Configuration File
Settings are stored in `model_config.json`:

```json
{
  "model_source": "github",
  "github_model_name": "final_model.pt",
  "local_model_path": "models/local_model.pt",
  "custom_model_url": "",
  "cache_duration_hours": 24,
  "auto_update": true,
  "fallback_to_default": true,
  "last_update_check": "2024-01-01T12:00:00",
  "model_versions": {}
}
```

### Configuration Options

- `model_source`: "github", "local", or "custom"
- `github_model_name`: Name of the model file in GitHub
- `cache_duration_hours`: How long to cache models (default: 24 hours)
- `auto_update`: Automatically check for GitHub updates
- `fallback_to_default`: Use default YOLOv5 if model loading fails

## CLI Usage

### Load Models

```bash
# Load from GitHub (default)
python model_cli.py load

# Load specific GitHub model
python model_cli.py load --source github --path "custom_model.pt"

# Load local model
python model_cli.py load --source local --path "models/my_model.pt"

# Force download from GitHub
python model_cli.py load --force
```

### List Available Models

```bash
python model_cli.py list
```

### Model Information

```bash
python model_cli.py info
```

### Cache Management

```bash
# Show cache information
python model_cli.py cache info

# Clear cache
python model_cli.py cache clear
```

### Configuration Management

```bash
# Show current configuration
python model_cli.py config show

# Switch to local model
python model_cli.py config --source local --local-path "models/my_model.pt"

# Set GitHub model name
python model_cli.py config --model-name "my_custom_model.pt"

# Disable auto-update
python model_cli.py config --no-auto-update
```

## REST API Endpoints

### Model Management

- `GET /model-manager/available-models` - List available GitHub models
- `POST /model-manager/switch-model` - Switch model source
- `GET /model-manager/config` - Get configuration
- `POST /model-manager/config` - Update configuration
- `GET /model-manager/cache-info` - Get cache information
- `POST /model-manager/clear-cache` - Clear cache

### Example API Usage

```javascript
// Switch to local model
fetch('/model-manager/switch-model', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        source: 'local',
        model_path: 'models/my_model.pt'
    })
});

// Get available models
fetch('/model-manager/available-models')
    .then(response => response.json())
    .then(data => console.log(data.models));
```

## File Structure

```
fl/
├── model_manager.py          # Main ModelManager class
├── model_cli.py             # Command-line interface
├── federated_app.py         # Updated Flask app
├── model_config.json        # Configuration file (auto-created)
├── model_cache/             # Cached models directory (auto-created)
│   ├── final_model.pt
│   └── ...
└── MODEL_MANAGEMENT.md      # This documentation
```

## Caching System

### How It Works

1. **First Load**: Downloads model from GitHub and caches locally
2. **Subsequent Loads**: Uses cached version if still valid
3. **Update Check**: Periodically checks GitHub for updates
4. **Version Tracking**: Uses file hashes to detect changes

### Cache Duration

- Default: 24 hours
- Configurable via `cache_duration_hours`
- Force refresh with `force_download=True`

### Cache Location

- Directory: `model_cache/`
- Configurable via ModelManager constructor
- Auto-created if doesn't exist

## Error Handling

### Fallback Strategy

1. Try to load from configured source
2. If GitHub fails, try cached version
3. If all fails, load default YOLOv5 model
4. Log all errors for debugging

### Common Issues

**Model Loading Fails**:
- Check internet connection for GitHub models
- Verify local model path exists
- Check model file format compatibility

**Cache Issues**:
- Clear cache: `python model_cli.py cache clear`
- Check disk space
- Verify file permissions

## Future Features

### Planned Enhancements

1. **User Training Interface**: Web UI for training custom models
2. **Model Versioning**: Support for multiple model versions
3. **Performance Metrics**: Model comparison and benchmarking
4. **Automated Testing**: Model validation on test datasets
5. **Cloud Integration**: Support for cloud model storage

### User Training Workflow (Future)

1. User uploads training data via web interface
2. System trains model locally using federated learning
3. User can save trained model locally
4. Option to upload to GitHub for sharing
5. Seamless switching between user models and GitHub models

## Troubleshooting

### Model Not Loading

```bash
# Check model info
python model_cli.py info

# Check available models
python model_cli.py list

# Clear cache and retry
python model_cli.py cache clear
python model_cli.py load --force
```

### Configuration Issues

```bash
# Reset to defaults
rm model_config.json
python model_cli.py load
```

### GitHub Access Issues

```bash
# Check network connectivity
curl -I https://raw.githubusercontent.com/KaitlynKK/fed-learning7-belalandkaitlyn/main/static/output/

# Use local model as fallback
python model_cli.py config --source local --local-path "models/fallback.pt"
```

## Support

For issues or questions:
1. Check the logs in the console output
2. Use the CLI tools for debugging
3. Check the cache and configuration files
4. Verify GitHub repository access
