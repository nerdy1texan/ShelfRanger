"""
GPU-Optimized Streamlit UI for ShelfRanger
Uses GPU acceleration when available, with proper CUDA handling.
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
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize GPU context early
@st.cache_resource
def initialize_torch():
    """Initialize PyTorch with proper CUDA handling."""
    try:
        import torch
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Initialize CUDA context
            torch.cuda.init()
            torch.cuda.empty_cache()
            
            return {
                'torch': torch,
                'device': 'cuda',
                'gpu_name': gpu_name,
                'gpu_memory': f"{gpu_memory:.1f}GB",
                'cuda_available': True
            }
        else:
            return {
                'torch': torch,
                'device': 'cpu',
                'gpu_name': 'CPU Only',
                'gpu_memory': 'N/A',
                'cuda_available': False
            }
            
    except Exception as e:
        st.error(f"❌ PyTorch initialization failed: {str(e)}")
        return None

# Initialize PyTorch
torch_info = initialize_torch()
if torch_info is None:
    st.stop()

# Import YOLO after PyTorch initialization
try:
    from ultralytics import YOLO
except Exception as e:
    st.error(f"❌ YOLO import failed: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ShelfRanger - GPU Accelerated Detection",
    page_icon="🚀",
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
.gpu-info {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    text-align: center;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_gpu(model_path, device):
    """Load YOLOv8 model with GPU optimization."""
    try:
        if not Path(model_path).exists():
            st.error(f"❌ Model not found: {model_path}")
            return None
        
        # Load model
        model = YOLO(model_path)
        
        # Move to appropriate device
        if hasattr(model, 'model') and device == 'cuda':
            model.model.to(device)
        
        # Test inference
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _ = model(test_img, device=device, verbose=False)
        
        return model
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

def run_detection_gpu(model, image, conf_threshold, device):
    """Run GPU-accelerated detection."""
    try:
        img_array = np.array(image)
        
        # Run inference on GPU/CPU
        results = model(img_array, conf=conf_threshold, device=device, verbose=False)
        
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
                color = (0, 255, 0)  # Green
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image, detections
        
    except Exception as e:
        st.error(f"❌ Detection failed: {str(e)}")
        return None, []

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 ShelfRanger GPU Detection</h1>', 
                unsafe_allow_html=True)
    
    # GPU Info Banner
    gpu_status = "🚀 GPU Accelerated" if torch_info['cuda_available'] else "🖥️ CPU Mode"
    st.markdown(f'''
    <div class="gpu-info">
        <h3>{gpu_status}</h3>
        <p><strong>Device:</strong> {torch_info['gpu_name']} | <strong>Memory:</strong> {torch_info['gpu_memory']}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ GPU Settings")
        
        # Device info
        st.subheader("🔧 Hardware Info")
        st.info(f"**Device:** {torch_info['device'].upper()}")
        if torch_info['cuda_available']:
            st.success(f"**GPU:** {torch_info['gpu_name']}")
            st.info(f"**VRAM:** {torch_info['gpu_memory']}")
        
        # Model selection
        st.subheader("🎯 Model Selection")
        model_options = []
        if Path("models/train/weights/best.pt").exists():
            model_options.append("models/train/weights/best.pt")
        if Path("models/train/weights/last.pt").exists():
            model_options.append("models/train/weights/last.pt")
        if Path("yolov8n.pt").exists():
            model_options.append("yolov8n.pt")
        
        if not model_options:
            st.error("❌ No models found")
            st.info("Run training first!")
            st.stop()
        
        model_path = st.selectbox("Select Model", model_options)
        
        # Detection settings
        st.subheader("🎛️ Detection Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        
        # Batch processing option
        if torch_info['cuda_available']:
            batch_processing = st.checkbox("Enable Batch Processing", value=False,
                                         help="Process multiple images faster")
        
        # Performance metrics
        st.subheader("📊 Model Metrics")
        results_path = Path("models/train/results.csv")
        if results_path.exists():
            try:
                df = pd.read_csv(results_path)
                final_metrics = df.iloc[-1]
                st.metric("mAP50", f"{final_metrics.get('metrics/mAP50(B)', 0):.3f}")
                st.metric("mAP50-95", f"{final_metrics.get('metrics/mAP50-95(B)', 0):.3f}")
                st.metric("Training Epochs", len(df))
            except:
                st.info("Metrics unavailable")
    
    # Load model
    device = torch_info['device']
    model = load_model_gpu(model_path, device)
    if model is None:
        st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Image Input")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Images", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True,
            help="Upload one or more images for detection"
        )
        
        # Example images
        st.subheader("📷 Example Images")
        dataset_path = Path("dataset/images/train")
        if dataset_path.exists():
            examples = list(dataset_path.glob("*.jpg"))[:5]
            if examples:
                selected = st.selectbox("Try example", ["None"] + [img.name for img in examples])
                if selected != "None":
                    example_file = open(dataset_path / selected, "rb")
                    uploaded_files = [example_file]
    
    with col2:
        st.header("🎯 Detection Results")
        
        if uploaded_files:
            # Process images
            for idx, uploaded_file in enumerate(uploaded_files):
                if uploaded_file is not None:
                    try:
                        # Load image
                        image = Image.open(uploaded_file).convert('RGB')
                        
                        # Show image info
                        st.subheader(f"Image {idx + 1}: {getattr(uploaded_file, 'name', 'Unknown')}")
                        
                        # Create columns for input and output
                        img_col1, img_col2 = st.columns(2)
                        
                        with img_col1:
                            st.write("**Input**")
                            st.image(image, use_column_width=True)
                        
                        # Run detection
                        with st.spinner(f"🚀 {'GPU' if device == 'cuda' else 'CPU'} Detection..."):
                            import time
                            start_time = time.time()
                            
                            annotated_img, detections = run_detection_gpu(
                                model, image, conf_threshold, device
                            )
                            
                            inference_time = time.time() - start_time
                        
                        with img_col2:
                            st.write("**Detections**")
                            if annotated_img is not None:
                                st.image(annotated_img, use_column_width=True)
                        
                        # Results summary
                        if detections:
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Detections", len(detections))
                            with col_b:
                                avg_conf = np.mean([d['confidence'] for d in detections])
                                st.metric("Avg Confidence", f"{avg_conf:.3f}")
                            with col_c:
                                st.metric("Inference Time", f"{inference_time:.3f}s")
                            
                            # Detailed results
                            with st.expander(f"📋 Detailed Results ({len(detections)} items)"):
                                df = pd.DataFrame([
                                    {
                                        'Product': d['product'],
                                        'Confidence': f"{d['confidence']:.3f}",
                                        'BBox': f"({d['bbox'][0]:.0f}, {d['bbox'][1]:.0f}, {d['bbox'][2]:.0f}, {d['bbox'][3]:.0f})"
                                    } for d in detections
                                ])
                                st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No products detected")
                        
                        st.divider()
                        
                    except Exception as e:
                        st.error(f"Error processing image {idx + 1}: {str(e)}")
        else:
            st.info("👆 Upload images to start GPU-accelerated detection")
    
    # Footer
    st.markdown("---")
    device_info = f"🚀 GPU: {torch_info['gpu_name']}" if torch_info['cuda_available'] else "🖥️ CPU Mode"
    st.markdown(f"🛒 **ShelfRanger** - {device_info} | Powered by YOLOv8")

if __name__ == "__main__":
    main() 