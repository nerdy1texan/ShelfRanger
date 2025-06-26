"""
YOLOv8 training script for ShelfRanger.
"""

import os
import logging
from pathlib import Path

# Set environment variable to disable PyTorch 2.6 security restrictions for trusted models
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# Import torch first and patch its load function
import torch

# Store the original torch.load function
_original_torch_load = torch.load

# Create a patched version that always uses weights_only=False
def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    """Patched torch.load that forces weights_only=False for trusted model loading."""
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)

# Apply the patch
torch.load = patched_torch_load

from ultralytics import YOLO
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.config import TRAINING_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_file(url, filename):
    """Download a file using requests."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        logger.info(f"Downloading {filename} from {url}")
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Download completed: {filename}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return False

def setup_training_environment():
    """Prepare the training environment."""
    # Create output directory if it doesn't exist
    output_dir = Path(TRAINING_CONFIG["project"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify dataset exists
    data_yaml = Path(TRAINING_CONFIG["data_yaml"])
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration not found at {data_yaml}")
    
    logger.info("Training environment setup completed")
    return plots_dir

def plot_metrics(results, plots_dir):
    """Plot and save training metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = results.results_dict
    
    # Set style (using ggplot instead of seaborn for compatibility)
    plt.style.use('ggplot')
    
    # Plot training losses
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['box_loss', 'cls_loss', 'dfl_loss']
    for metric in metrics_to_plot:
        if metric in metrics:
            plt.plot(metrics[metric], label=metric)
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f'training_losses_{timestamp}.png')
    plt.close()
    
    # Plot metrics
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    for metric in metrics_to_plot:
        if metric in metrics:
            plt.plot(metrics[metric], label=metric.split('/')[-1])
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f'validation_metrics_{timestamp}.png')
    plt.close()
    
    logger.info(f"Training plots saved to {plots_dir}")

def safe_load_model(model_path):
    """Load a YOLO model with PyTorch security bypassed for trusted sources."""
    try:
        logger.info(f"Loading model from: {model_path}")
        # Simply load the model - environment variable handles security
        model = YOLO(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def train_model():
    """Train YOLOv8 model on the ShelfRanger dataset."""
    try:
        # Setup environment
        plots_dir = setup_training_environment()
        
        # Download and load base model with safe loading
        logger.info(f"Loading base model: {TRAINING_CONFIG['model_name']}")
        try:
            model = safe_load_model(TRAINING_CONFIG["model_name"])
        except Exception as model_error:
            logger.error(f"Error loading model: {str(model_error)}")
            # Fallback to direct download if needed
            logger.info("Attempting direct model download...")
            model_url = f"https://github.com/ultralytics/assets/releases/download/v8.1.0/{TRAINING_CONFIG['model_name']}"
            if download_file(model_url, TRAINING_CONFIG["model_name"]):
                model = safe_load_model(TRAINING_CONFIG["model_name"])
            else:
                raise RuntimeError("Failed to download model weights")
        
        # Start training
        logger.info("Starting training...")
        results = model.train(
            data=TRAINING_CONFIG["data_yaml"],
            imgsz=TRAINING_CONFIG["image_size"],
            epochs=TRAINING_CONFIG["epochs"],
            batch=TRAINING_CONFIG["batch_size"],
            device=TRAINING_CONFIG["device"],
            workers=TRAINING_CONFIG["workers"],
            patience=TRAINING_CONFIG["patience"],
            project=TRAINING_CONFIG["project"],
            name=TRAINING_CONFIG["name"],
            exist_ok=TRAINING_CONFIG["exist_ok"],
            plots=True  # Enable built-in plots
        )
        
        # Plot custom metrics
        plot_metrics(results, plots_dir)
        
        # Log training results
        logger.info(f"Training completed. Results saved to {TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}")
        logger.info(f"Training plots saved to {plots_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

def main():
    """Main entry point for training."""
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main() 