# Crop Growth Stage Detection using Computer Vision

## Overview

This project is an AI-powered agricultural monitoring system that detects crop growth stages using advanced Computer Vision and Deep Learning techniques. 

The system combines:
- YOLOv8 Object Detection
- ResNet50 Transfer Learning
- Custom CNN Classification
- A Clean, Production-Ready Streamlit Interface

We have streamlined the application to focus heavily on fast, reliable, and user-friendly real-time crop image classification, ensuring a premium experience.

---

# Crop Growth Stages Detected

The system detects the following crop growth stages:

- Germination
- Vegetative
- Flowering
- Harvesting

---

# Features

## Deep Learning Models
- **YOLOv8** for fast object detection and classification
- **ResNet50** for robust transfer learning feature extraction
- **Custom CNN** for lightweight, CPU-friendly image classification

## Interactive Streamlit Web Application
- **Clean UI**: A completely redesigned, premium layout focusing exclusively on prediction results.
- **Model Selection**: Dynamically switch between YOLOv8, ResNet50, and Custom CNN using a sidebar dropdown.
- **Image Upload & Previews**: Upload your custom crop images to run inference on-the-fly.
- **Built-in Examples**: Test the system immediately using an anonymous gallery of built-in examples.
- **Lazy Loading**: Models are loaded lazily and cached using Streamlit's `@st.cache_resource`, ensuring ultra-fast app startup and efficient memory usage.

---

# Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- PyTorch
- Torchvision
- Streamlit
- NumPy
- PIL (Python Imaging Library)

---

# Project Workflow

```text
User Selects/Uploads Crop Image
        ↓
Select Deep Learning Model (YOLOv8 / CNN / ResNet50)
        ↓
Submit for Analysis
        ↓
Inference (with Cached Models)
        ↓
Classification Output & Confidence Score Display
```

---

# Project Structure

```bash
crop-growth-stage-detection/

│
├── app.py                 # Main Streamlit web application
├── requirements.txt       # Optimized dependencies
├── README.md
├── .gitignore
│
├── models/                # Trained deep learning models
│   ├── YOLO.pt
│   ├── custom_cnn_model.pth
│   └── resnet_model.pth
│
├── examples/              # Built-in test images
│   ├── germination.jpg
│   ├── vegetative.jpg
│   ├── flowering.jpg
│   └── harvesting.jpg
│
└── utils/
    └── prediction.py      # Core prediction logic & lazy model loaders
```

---

# Installation & Setup Guide

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/crop-growth-stage-detection.git
cd crop-growth-stage-detection
```

## Step 2: Create Virtual Environment

```bash
python -m venv venv
```

## Step 3: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```
**Linux / Mac:**
```bash
source venv/bin/activate
```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Run Streamlit Application

```bash
streamlit run app.py
```

The application will launch automatically in your browser at `http://localhost:8501`.

---

# Deep Learning Models Used

### 1. YOLOv8
YOLOv8 is utilized for classification leveraging its localized regional characteristics.
- High-speed detection
- Anchor-free architecture
- Real-time performance

### 2. Custom CNN
The custom 3-layer convolutional network is built for fast, CPU-friendly classification.
- Convolution and Max pooling layers
- Dropout regularization
- Dense layers & Softmax classifier

### 3. ResNet50
An industry-standard deep residual network loaded with custom transfer-learning weights.
- Better feature extraction
- Deep residual architecture
- High accuracy generalizations

---

# Author

- Vikas Kumar Yadav

---

# License

This project is developed for educational and research purposes.