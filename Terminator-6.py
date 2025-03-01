# Check if required libraries are installed and install them if not
import importlib.util

def check_and_install(package):
    spec = importlib.util.find_spec(package)
    if spec is None:
        print(f"{package} not found. Installing...")
        !pip install {package}
    else:
        print(f"{package} already installed.")

# List of required libraries
required_packages = [
    'torch',
    'torchvision',
    'scikit-learn',
    'faiss-cpu',
    'pyautogui',
    'Pillow',
    'opencv-python',
    'loguru'
]

for package in required_packages:
    check_and_install(package)

# Import necessary libraries
import os
import cv2
from sklearn.preprocessing import normalize
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pyautogui

# Data Augmentation and Transformation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and augment the dataset
dataset = datasets.ImageFolder('path_to_your_dataset', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# Custom Feature Extraction Model
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, 256)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Initialize the model
num_classes = len(dataset.classes)
model = CustomModel(num_classes).cuda()
model.eval()

# Load pre-trained weights (if available)
pretrained_weights_path = 'path_to_pretrained_weights.pth'
if os.path.exists(pretrained_weights_path):
    model.load_state_dict(torch.load(pretrained_weights_path))
    print(f"Loaded pre-trained weights from {pretrained_weights_path}")

# Extract features from the augmented dataset
def extract_features(model, dataloader):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.cuda()
            features = model(images).cpu().numpy()
            all_features.append(features)
            all_labels.extend(labels.numpy())
    return np.concatenate(all_features), np.array(all_labels)

features, labels = extract_features(model, dataloader)

# Initialize FAISS index with L2 distance and advanced indexing
index = faiss.IndexFlatL2(features.shape[1])
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.add(features)  # Add database features to the GPU index

# Function to find the closest feature vector in the database
def find_closest_feature(target_feature):
    D, I = gpu_index.search(target_feature, k=1)  # Search for the top nearest vector
    return I[0][0], D[0][0]  # Return the index and distance of the closest feature

# Object Detection using a pre-trained model (e.g., Faster R-CNN)
def detect_objects(image_path):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    obj_model = fasterrcnn_resnet50_fpn(weights=weights).cuda()
    obj_model.eval()

    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).cuda()  # Add batch dimension

    with torch.no_grad():
        predictions = obj_model(img_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    return boxes, scores

# Function to simulate locking onto the target using mouse and keyboard
def lock_on_target(boxes, scores, image_path):
    if len(boxes) == 0:
        logger.warning(f"No targets detected in {image_path}.")
        return

    # For demonstration, let's assume we are interested in the highest scoring box
    best_idx = np.argmax(scores)
    x1, y1, x2, y2 = boxes[best_idx]

    # Calculate center coordinates of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Move the mouse to the center of the detected target
    screen_width, screen_height = pyautogui.size()
    img_width, img_height = Image.open(image_path).size
    scale_x = screen_width / img_width
    scale_y = screen_height / img_height

    scaled_center_x = center_x * scale_x
    scaled_center_y = center_y * scale_y

    try:
        pyautogui.moveTo(scaled_center_x, scaled_center_y, duration=0.5)
        logger.info(f"Locking onto target at ({scaled_center_x}, {scaled_center_y}) in {image_path}")
    except Exception as e:
        logger.error(f"Error moving mouse for image {image_path}: {e}")

# Function to process a new image
def process_new_image(image_path):
    try:
        boxes, scores = detect_objects(image_path)

        # Find the closest feature in the database (optional)
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).cuda()  # Add batch dimension

        with torch.no_grad():
            target_feature = model(img_tensor).cpu().numpy()

        closest_index, distance = find_closest_feature(target_feature)
        logger.info(f"Closest feature index for {image_path}: {closest_index} with distance {distance}")

        # Lock onto the target
        lock_on_target(boxes, scores, image_path)
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")

# Use multi-threading to process multiple new images concurrently
new_image_paths = ['path_to_new_image1.jpg', 'path_to_new_image2.jpg']  # Add your image paths here

from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_new_image, new_image_paths)

logger.info("Automatic lock-on targeting complete.")
