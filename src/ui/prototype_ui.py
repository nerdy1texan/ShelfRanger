"""
Streamlit UI for ShelfRanger prototype.
"""

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from prototype_sku_lookup import SKU_LOOKUP
from utils.visualization import draw_detections, create_results_summary

# Set page config
st.set_page_config(
    page_title="ShelfRanger - SKU Detection Prototype",
    page_icon="🔍",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🔍 ShelfRanger")
    st.subheader("SKU Detection Prototype")
    
    # Model selection
    model_path = st.selectbox(
        "Select Model",
        ["models/train/weights/best.pt", "yolov8n.pt"],
        help="Select the trained model or use the base model"
    )
    
    # Confidence threshold
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        help="Minimum confidence score for detection"
    )
    
    st.markdown("---")
    st.markdown("""
    ### Instructions
    1. Upload an image
    2. Wait for detection
    3. View results in the main panel
    
    ### Training Command
    ```bash
    python src/training/train.py
    ```
    
    ### Run UI
    ```bash
    streamlit run src/ui/prototype_ui.py
    ```
    """)

# Main content
st.title("📦 Product Detection & SKU Lookup")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to numpy array for YOLO
        image_array = np.array(image)
        
        # Load model and run inference
        try:
            model = YOLO(model_path)
            results = model.predict(image_array, conf=conf_threshold)[0]
            
            # Get annotated image
            annotated_img = results.plot()
            st.image(annotated_img, caption="Detection Results", use_column_width=True)
            
        except Exception as e:
            st.error(f"Error running detection: {str(e)}")
            st.info("Make sure you have trained the model and the weights file exists at the specified path.")
    
    with col2:
        st.subheader("Detection Results")
        
        if 'results' in locals():
            # Create detection table
            detections = []
            
            # Process each detection
            for box in results.boxes:
                class_id = results.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                # Get product info from SKU_LOOKUP
                product_info = SKU_LOOKUP.get(class_id, {
                    "product_name": "Unknown",
                    "location": "Unknown"
                })
                
                detections.append({
                    "sku_id": class_id,
                    "product_name": product_info["product_name"],
                    "location": product_info["location"],
                    "confidence": f"{confidence:.2%}"
                })
            
            # Display results table
            if detections:
                st.dataframe(
                    detections,
                    column_config={
                        "sku_id": "SKU ID",
                        "product_name": "Product Name",
                        "location": "Store Location",
                        "confidence": "Confidence"
                    },
                    hide_index=True
                )
                
                # Show raw JSON
                with st.expander("Show Raw Detection Data"):
                    st.json(json.dumps(detections, indent=2))
            else:
                st.warning("No products detected in the image.") 