# 🔍 ShelfRanger - Retail Shelf Monitoring System

## 📦 Project Overview
ShelfRanger is an intelligent retail shelf monitoring and automated restocking system that uses YOLOv8 for product detection and classification.

## 🏗️ Architecture
![ShelfRanger Architecture](ShelfRanger%20Architecture.png)

## 🏗️ Project Structure
```
ShelfRanger/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── environment.yml            # Conda environment specification
├── src/
│   ├── training/             # Model training code
│   │   ├── __init__.py
│   │   ├── train.py         # YOLOv8 training script
│   │   └── config.py        # Training configuration
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   └── visualization.py # Visualization helpers
│   └── ui/                  # Streamlit UI code
│       ├── __init__.py
│       └── prototype_ui.py  # Main Streamlit interface
├── models/                   # Trained model weights
│   └── .gitkeep
├── dataset/                 # YOLOv8 dataset
│   ├── data.yaml           # Dataset configuration
│   ├── train/              # Training images and labels
│   ├── valid/              # Validation images and labels
│   └── test/               # Test images and labels
└── prototype_sku_lookup.py  # SKU mapping dictionary
```

## 💻 Technology Stack
![Tech Stack 1](Tech%20Stack%201.png)
![Tech Stack 2](Tech%20Stack%20%202.png)

## 🚀 Setup Instructions

### 1. Create Conda Environment
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate shelfranger
# Activate environment git bash
source /c/Users/mauli/anaconda3/Scripts/activate shelfranger

# Update Environment
conda env update -f environment.yml --prune

# Clean up and re-train from scratch
rm -rf models/* yolov8n.pt

```
### 2. Train YOLOv8 Model
```bash
# From project root
python src/training/train.py
```

### 3. Run Streamlit UI
```bash
# From project root
streamlit run src/ui/prototype_ui.py
```

## 📝 Model Training Details
- Base model: YOLOv8n
- Image size: 512x512
- Epochs: 50
- Dataset: Custom retail shelf dataset with 39 SKU classes
- Training script: `src/training/train.py`

## 📈 Training Results
![Training Results](Training%20Results.png)
![Training Results 2](Training%20Results%202.png)

## 🛠️ Dependencies
See `requirements.txt` for Python package dependencies.

## 📊 Dataset Structure
- Training: 2,793 images
- Validation: 264 images
- Test: 100 images
- Classes: 39 SKUs (see data.yaml for full list) # ShelfRanger
