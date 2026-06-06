import streamlit as st
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import numpy as np
import cv2

# =========================
# CLASS NAMES
# =========================
classes = [
    "Flowering",
    "Germination",
    "Harvesting",
    "Vegetative"
]

# =========================
# DEVICE
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =========================
# CNN MODEL ARCHITECTURE
# =========================
class CropCNN(nn.Module):
    def __init__(self):
        super(CropCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =========================
# LAZY MODEL CACHING
# =========================
@st.cache_resource
def load_yolo_model():
    return YOLO("models/YOLO.pt")

@st.cache_resource
def load_cnn_model():
    model = CropCNN().to(device)
    model.load_state_dict(
        torch.load(
            "models/custom_cnn_model.pth",
            map_location=device
        )
    )
    model.eval()
    return model

@st.cache_resource
def load_resnet_model():
    try:
        model = models.resnet50(weights=None)
    except TypeError:
        model = models.resnet50(pretrained=False)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)
    model.load_state_dict(
        torch.load(
            "models/resnet_model.pth",
            map_location=device
        )
    )
    model = model.to(device)
    model.eval()
    return model

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# PREDICTION FUNCTIONS
# =========================
def predict_yolo(image):
    model = load_yolo_model()
    results = model(image, verbose=False)
    result = results[0]
    if len(result.boxes) > 0:
        class_id = int(result.boxes.cls[0])
        confidence = float(result.boxes.conf[0])
        return classes[class_id], confidence
    return "No Detection", 0.0

def predict_cnn(image):
    model = load_cnn_model()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
        
    return classes[predicted.item()], confidence

def predict_resnet(image):
    model = load_resnet_model()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
        
    return classes[predicted.item()], confidence
