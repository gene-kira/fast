import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pyautogui
import logging
import subprocess
import sys
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Rotate
from albumentations.pytorch import ToTensorV2 as ToTensor
import os
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Automatic library installation
required_packages = [
    'torch', 'torchvision', 'opencv-python', 'numpy', 'pyautogui', 'albumentations'
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install YOLOv5
if not os.path.exists("yolov5"):
    logger.info("Cloning YOLOv5 repository...")
    subprocess.check_call(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
os.chdir("yolov5")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Data Augmentation
def apply_augmentations(image):
    image_pil = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    augment = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        Rotate(limit=30, p=0.5)
    ])
    augmented = augment(image=image_pil)
    return cv2.cvtColor(np.array(augmented['image']), cv2.COLOR_RGB2BGR)

# Custom Dataset with Data Augmentation
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augmentations=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)  # Use OpenCV to read images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        if self.augmentations:
            image = apply_augmentations(image)
        
        if self.transform is not None:
            image = self.transform(image=np.array(image))['image']
        
        return image, label

# EfficientNet Feature Extraction
def extract_features(model, images):
    model.eval()
    with torch.no_grad():
        features = model(images).squeeze().numpy()
    return features

# Object Detection using YOLOv5 (for better accuracy)
from utils.general import non_max_suppression

def detect_objects_yolov5(model, image):
    results = model(image)[0]
    detections = non_max_suppression(results, 0.5, 0.4)  # Confidence and IoU thresholds
    return detections[0].cpu().numpy()

# Lock-on Target with Kalman Filter
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
    
    def update(self, bbox):
        if len(bbox) == 4:
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            measurement = np.array([[np.float32(x_center)], [np.float32(y_center)]])
            self.kf.correct(measurement)
            predicted = self.kf.predict()
            return predicted[0][0], predicted[1][0]
        return None, None

def lock_on_target(target_image, bbox, kalman_filter):
    if not bbox:
        logger.warning("No bounding box detected")
        return
    logger.info("Locking onto target")
    screen_width, screen_height = pyautogui.size()
    
    # Use Kalman Filter to predict the next position
    x_center_predicted, y_center_predicted = kalman_filter.update(bbox)
    
    if x_center_predicted is None or y_center_predicted is None:
        return
    
    screen_x = x_center_predicted * (screen_width / target_image.shape[1])
    screen_y = y_center_predicted * (screen_height / target_image.shape[0])
    
    pyautogui.moveTo(screen_x, screen_y, duration=0.1)  # Smooth movement
    logger.info(f"Mouse moved to ({screen_x:.2f}, {screen_y:.2f})")

# Fine-tune EfficientNet for custom task
def fine_tune_efficientnet(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Transform for images
transform = Compose([
    transforms.Resize((224, 224)),
    ToTensor()
])

# Initialize EfficientNet model
efficientnet_model = models.efficientnet_b0(pretrained=True)
num_ftrs = efficientnet_model.classifier[1].in_features
efficientnet_model.classifier[1] = torch.nn.Linear(num_ftrs, 2)  # Modify classifier for custom task

# Example image paths and labels (replace with your dataset)
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
labels = [0, 1]

dataset = CustomDataset(image_paths, labels, transform=transform, augmentations=True)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fine-tune EfficientNet (uncomment to train)
# fine_tune_efficientnet(efficientnet_model, train_loader, torch.nn.CrossEntropyLoss(), torch.optim.Adam(efficientnet_model.parameters()))

# Load YOLOv5 model
os.chdir("../")  # Move back to the root directory
from yolov5.models.experimental import attempt_load

yolo_model = attempt_load('yolov5s.pt', map_location='cpu')  # Use a smaller model for faster inference

# Initialize Kalman Filter
kalman_filter = KalmanFilter()

# Load EfficientNet classifier if trained
try:
    efficientnet_model.load_state_dict(torch.load("efficientnet_classifier.pth", map_location=torch.device('cpu')))
except FileNotFoundError:
    logger.warning("Pre-trained EfficientNet classifier not found. Using original model.")

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use default camera (change index if needed)

while True:
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to capture frame")
        break
    
    # Convert frame to RGB for YOLOv5 and EfficientNet
    frame_rgb_yolo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb_efficientnet = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects using YOLOv5
    detections = detect_objects_yolov5(yolo_model, [frame_rgb_yolo])
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:  # Confidence threshold
            bbox = (x1.item(), y1.item(), x2.item(), y2.item())
            
            # Prepare image for EfficientNet classification
            img = transforms.ToTensor()(frame_rgb_efficientnet)
            img = torch.unsqueeze(img, 0).float()
            with torch.no_grad():
                output = efficientnet_model(img)
                _, predicted = torch.max(output.data, 1)
                predicted_class = predicted.item()
            
            logger.info(f"Detected class: {predicted_class} with confidence {output[0][predicted_class]:.2f}")
            
            if predicted_class == 1:
                lock_on_target(frame, bbox, kalman_filter)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
