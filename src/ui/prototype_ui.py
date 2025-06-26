"""
Streamlit UI for ShelfRanger - Product Detection System
Test your trained YOLOv8 model with an interactive web interface.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
from pathlib import Path
import sys
import os
from datetime import datetime

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Set PyTorch environment for model loading
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# Handle PyTorch import with CUDA compatibility
try:
    import torch
    # Force CPU mode if CUDA has issues
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality
            test_tensor = torch.randn(1).cuda()
            cuda_available = True
        except Exception as e:
            st.warning(f"⚠️ CUDA detected but not functional in Streamlit. Using CPU mode. Error: {str(e)}")
            # Force CPU mode
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            cuda_available = False
    else:
        cuda_available = False
except Exception as e:
    st.error(f"❌ Error importing PyTorch: {str(e)}")
    st.info("💡 Try restarting the Streamlit server or reinstalling PyTorch")
    st.stop()

# Import YOLO after PyTorch setup
try:
    from ultralytics import YOLO
except Exception as e:
    st.error(f"❌ Error importing YOLO: {str(e)}")
    st.info("💡 Make sure ultralytics is installed: pip install ultralytics")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ShelfRanger - Product Detection",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}

.success-box {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #c3e6cb;
    margin: 1rem 0;
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
def load_model(model_path="models/train/weights/best.pt"):
    """Load the trained YOLOv8 model with caching."""
    try:
        if not Path(model_path).exists():
            st.error(f"❌ Model not found at {model_path}")
            st.info("💡 Make sure to run training first to generate the model weights.")
            return None
        
        # Load model and force CPU if CUDA has issues
        model = YOLO(model_path)
        
        # Force CPU mode for Streamlit compatibility
        if hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model.to('cpu')
        
        # Test model inference
        try:
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = model(test_img, device='cpu', verbose=False)
            st.success("✅ Model loaded and tested successfully")
        except Exception as test_error:
            st.warning(f"⚠️ Model loaded but inference test failed: {str(test_error)}")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def process_image(model, image, conf_threshold=0.25, iou_threshold=0.45):
    """Process image with YOLOv8 model and return results."""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run inference (force CPU for Streamlit compatibility)
        results = model(img_array, conf=conf_threshold, iou=iou_threshold, device='cpu')
        
        # Extract detection information
        detections = []
        annotated_image = img_array.copy()
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                class_name = model.names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': box.tolist()
                })
                
                # Draw bounding box
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image, detections
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, []

def create_detection_summary(detections):
    """Create a summary dataframe of detections."""
    if not detections:
        return pd.DataFrame()
    
    # Count detections by class
    class_counts = {}
    confidence_stats = {}
    
    for det in detections:
        class_name = det['class']
        confidence = det['confidence']
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            confidence_stats[class_name] = []
        
        class_counts[class_name] += 1
        confidence_stats[class_name].append(confidence)
    
    # Create summary dataframe
    summary_data = []
    for class_name, count in class_counts.items():
        confidences = confidence_stats[class_name]
        summary_data.append({
            'Product': class_name,
            'Count': count,
            'Avg Confidence': f"{np.mean(confidences):.3f}",
            'Max Confidence': f"{np.max(confidences):.3f}",
            'Min Confidence': f"{np.min(confidences):.3f}"
        })
    
    return pd.DataFrame(summary_data)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🛒 ShelfRanger Product Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">🎯 Upload an image to detect products using your trained YOLOv8 model!</div>', 
                unsafe_allow_html=True)
    
    # Sidebar for model info and settings
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        # Model selection
        model_path = st.selectbox(
            "Select Model",
            ["models/train/weights/best.pt", "models/train/weights/last.pt"],
            help="Choose between best (highest mAP) or last (final epoch) model"
        )
        
        # Detection parameters
        st.subheader("Detection Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05,
                                 help="Minimum confidence for detections")
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05,
                                help="IoU threshold for Non-Maximum Suppression")
        
        # Model info
        st.subheader("📊 Model Information")
        
        # Check if model exists and show stats
        if Path(model_path).exists():
            st.success("✅ Model loaded successfully")
            
            # Try to load results for model stats
            results_path = Path("models/train/results.csv")
            if results_path.exists():
                try:
                    df = pd.read_csv(results_path)
                    final_metrics = df.iloc[-1]
                    
                    st.metric("Final mAP50", f"{final_metrics.get('metrics/mAP50(B)', 0):.3f}")
                    st.metric("Final mAP50-95", f"{final_metrics.get('metrics/mAP50-95(B)', 0):.3f}")
                    st.metric("Training Epochs", len(df))
                except:
                    st.info("💡 Model metrics not available")
        else:
            st.error("❌ Model not found")
            st.info("🔄 Run training first to generate model weights")
    
    # Load model
    model = load_model(model_path)
    
    if model is None:
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        # Example images
        st.subheader("📷 Or try example images:")
        example_images = []
        dataset_path = Path("dataset/images/train")
        if dataset_path.exists():
            example_images = list(dataset_path.glob("*.jpg"))[:5]  # First 5 images
        
        if example_images:
            selected_example = st.selectbox(
                "Select example image",
                ["None"] + [img.name for img in example_images]
            )
            
            if selected_example != "None":
                example_path = dataset_path / selected_example
                uploaded_file = open(example_path, "rb")
    
    with col2:
        st.header("🎯 Detection Results")
        
        if uploaded_file is not None:
            # Process the image
            try:
                # Load image
                image = Image.open(uploaded_file).convert('RGB')
                
                # Display original image
                st.subheader("Original Image")
                st.image(image, caption="Input Image", use_column_width=True)
                
                # Process with model
                with st.spinner("🔍 Running detection..."):
                    annotated_image, detections = process_image(
                        model, image, conf_threshold, iou_threshold
                    )
                
                if annotated_image is not None:
                    # Display results
                    st.subheader("Detection Results")
                    st.image(annotated_image, caption="Detected Products", use_column_width=True)
                    
                    # Detection statistics
                    st.subheader("📊 Detection Summary")
                    
                    if detections:
                        # Metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Detections", len(detections))
                        with col_b:
                            unique_classes = len(set(det['class'] for det in detections))
                            st.metric("Unique Products", unique_classes)
                        with col_c:
                            avg_conf = np.mean([det['confidence'] for det in detections])
                            st.metric("Avg Confidence", f"{avg_conf:.3f}")
                        
                        # Detailed results table
                        summary_df = create_detection_summary(detections)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Individual detections
                        with st.expander("📋 Detailed Detection List"):
                            for i, det in enumerate(detections, 1):
                                st.write(f"**Detection {i}:**")
                                st.write(f"- Product: {det['class']}")
                                st.write(f"- Confidence: {det['confidence']:.3f}")
                                st.write(f"- Bounding Box: {[round(x, 1) for x in det['bbox']]}")
                                st.divider()
                        
                        # Download results
                        if st.button("📥 Download Results"):
                            # Create detailed results
                            results_data = {
                                'detection_id': range(1, len(detections) + 1),
                                'product_class': [det['class'] for det in detections],
                                'confidence': [det['confidence'] for det in detections],
                                'bbox_x1': [det['bbox'][0] for det in detections],
                                'bbox_y1': [det['bbox'][1] for det in detections],
                                'bbox_x2': [det['bbox'][2] for det in detections],
                                'bbox_y2': [det['bbox'][3] for det in detections],
                                'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(detections)
                            }
                            
                            results_df = pd.DataFrame(results_data)
                            csv = results_df.to_csv(index=False)
                            
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"shelf_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("🔍 No products detected. Try adjusting the confidence threshold or use a different image.")
                        
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        else:
            st.info("👆 Upload an image or select an example to start detection")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; margin-top: 2rem;">'
        '🛒 ShelfRanger - Powered by YOLOv8 | Built with Streamlit'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 