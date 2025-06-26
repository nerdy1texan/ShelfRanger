"""
Training configuration for YOLOv8 model.
"""

TRAINING_CONFIG = {
    # Model parameters
    "model_name": "yolov8n.pt",  # Base model
    "image_size": 512,           # Input image size
    "epochs": 50,               # Number of training epochs
    
    # Dataset parameters
    "data_yaml": "dataset/data.yaml",  # Dataset configuration
    "batch_size": 32,           # Batch size (optimized for GPU)
    
    # Training parameters
    "device": 0,                # Training device (GPU 0 - RTX 3060)
    "workers": 4,               # Number of worker threads
    "patience": 50,             # Early stopping patience
    
    # Output parameters
    "project": "models",        # Project directory
    "name": "train",            # Run name
    "exist_ok": True,           # Overwrite existing files
}

# Validation parameters
VALIDATION_CONFIG = {
    "conf": 0.25,              # Confidence threshold
    "iou": 0.45,               # NMS IoU threshold
} 