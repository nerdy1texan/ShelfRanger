"""
Visualization utilities for ShelfRanger.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple

def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw detection boxes and labels on image.
    
    Args:
        image: Input image as numpy array
        detections: List of detection dictionaries with 'box', 'confidence', and 'class_name'
        color: Box color in BGR format
        thickness: Line thickness
    
    Returns:
        Annotated image
    """
    img = image.copy()
    
    for det in detections:
        # Extract box coordinates
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        label = f"{det['class_name']} {conf:.2f}"
        
        # Draw box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (int(x1), int(y1) - 20), (int(x1) + label_w, int(y1)), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img

def create_results_summary(
    detections: List[Dict],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Create a visual summary of detection results.
    
    Args:
        detections: List of detection dictionaries
        image_shape: Original image dimensions (height, width)
    
    Returns:
        Summary visualization as numpy array
    """
    # Create blank canvas
    height, width = image_shape
    summary = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw detection statistics
    y_pos = 30
    cv2.putText(summary, f"Total Detections: {len(detections)}", 
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # List each unique class and count
    class_counts = {}
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    y_pos += 40
    for class_name, count in class_counts.items():
        cv2.putText(summary, f"{class_name}: {count}", 
                    (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        y_pos += 30
    
    return summary 