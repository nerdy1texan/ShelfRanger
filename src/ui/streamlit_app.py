"""
Simplified Streamlit UI for ShelfRanger - CPU Only Version
Test your trained YOLOv8 model with CPU inference for maximum compatibility.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import sys
import os
from datetime import datetime

# Force CPU mode to avoid CUDA issues in Streamlit
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import YOLO with error handling
try:
    from ultralytics import YOLO
    import torch
    # Verify CPU mode
    st.info(f"🖥️ Running in CPU mode for maximum compatibility")
except Exception as e:
    st.error(f"❌ Error importing libraries: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ShelfRanger - Product Detection (CPU)",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
}
.info-box {
    background-color: #d1ecf1;
    color: #0c5460;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #bee5eb;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_cpu(model_path="models/train/weights/best.pt"):
    """Load YOLOv8 model in CPU mode."""
    try:
        if not Path(model_path).exists():
            st.error(f"❌ Model not found: {model_path}")
            return None
        
        # Load model
        model = YOLO(model_path)
        
        # Test inference
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _ = model(test_img, device='cpu', verbose=False)
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def detect_products(model, image, conf_threshold=0.25):
    """Run detection on image using CPU."""
    try:
        img_array = np.array(image)
        
        # Run inference
        results = model(img_array, conf=conf_threshold, device='cpu', verbose=False)
        
        # Process results
        detections = []
        annotated_image = img_array.copy()
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                class_name = model.names[cls]
                
                detections.append({
                    'product': class_name,
                    'confidence': conf,
                    'bbox': box.tolist()
                })
                
                # Draw detection
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_image, detections
        
    except Exception as e:
        st.error(f"❌ Detection error: {str(e)}")
        return None, []

def main():
    # Header
    st.markdown('<h1 class="main-header">🛒 ShelfRanger Product Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">🖥️ CPU-optimized version for maximum compatibility</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_options = []
        if Path("models/train/weights/best.pt").exists():
            model_options.append("models/train/weights/best.pt")
        if Path("models/train/weights/last.pt").exists():
            model_options.append("models/train/weights/last.pt")
        if Path("yolov8n.pt").exists():
            model_options.append("yolov8n.pt")
        
        if not model_options:
            st.error("❌ No models found")
            st.info("Run training first: `python src/training/train.py`")
            st.stop()
        
        model_path = st.selectbox("Select Model", model_options)
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        
        # Model info
        st.subheader("📊 Model Info")
        results_path = Path("models/train/results.csv")
        if results_path.exists():
            try:
                df = pd.read_csv(results_path)
                final_metrics = df.iloc[-1]
                st.metric("mAP50", f"{final_metrics.get('metrics/mAP50(B)', 0):.3f}")
                st.metric("Epochs", len(df))
            except:
                st.info("Metrics not available")
    
    # Load model
    model = load_model_cpu(model_path)
    if model is None:
        st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg'])
        
        # Example images
        dataset_path = Path("dataset/images/train")
        if dataset_path.exists():
            examples = list(dataset_path.glob("*.jpg"))[:3]
            if examples:
                st.subheader("📷 Examples")
                selected = st.selectbox("Try example", ["None"] + [img.name for img in examples])
                if selected != "None":
                    uploaded_file = open(dataset_path / selected, "rb")
    
    with col2:
        st.header("🎯 Results")
        
        if uploaded_file is not None:
            try:
                # Load and display image
                image = Image.open(uploaded_file).convert('RGB')
                st.subheader("Input")
                st.image(image, use_column_width=True)
                
                # Run detection
                with st.spinner("🔍 Detecting..."):
                    annotated_img, detections = detect_products(model, image, conf_threshold)
                
                if annotated_img is not None:
                    st.subheader("Detections")
                    st.image(annotated_img, use_column_width=True)
                    
                    if detections:
                        # Summary
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Total Detections", len(detections))
                        with col_b:
                            avg_conf = np.mean([d['confidence'] for d in detections])
                            st.metric("Avg Confidence", f"{avg_conf:.3f}")
                        
                        # Details table
                        df = pd.DataFrame([
                            {
                                'Product': d['product'],
                                'Confidence': f"{d['confidence']:.3f}"
                            } for d in detections
                        ])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No products detected")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.info("Upload an image to start detection")
    
    # Footer
    st.markdown("---")
    st.markdown("🛒 **ShelfRanger** - Product Detection System")

if __name__ == "__main__":
    main() 