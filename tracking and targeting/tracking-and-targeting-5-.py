import pygame
import cv2
import sys
import os
import subprocess
import numpy as np
import torch
from torchvision.transforms import functional as F
from ultralytics import YOLO
from collections import deque
import logging
from torch.nn import Transformer, TransformerEncoderLayer
import torch.optim as optim

# Function to check and install required libraries
def ensure_installed(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ["pygame", "opencv-python-headless", "numpy", "torch", "ultralytics"]
for package in required_packages:
    ensure_installed(package)

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Production-Ready Mouse Tracking and Targeting")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Webcam setup with higher resolution
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set resolution to 1280x720 if supported
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Load pre-trained model for object detection
model_detection = YOLO('yolov8n.pt')

def detect_objects(image):
    results = model_detection.predict(image, conf=0.5)
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            detections.append((r[0], r[1], r[2] - r[0], r[3] - r[1]))
    return detections

# Target variables
target_radius = 3  # 3mm threshold, adjust based on screen resolution
locked_target = None

# Transformer model for predictive tracking
history_length = 10
x_history = deque(maxlen=history_length)
y_history = deque(maxlen=history_length)

# Define the Transformer model
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerPredictor(torch.nn.Module):
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=6, dim_feedforward=256, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.linear_out = torch.nn.Linear(d_model, 2)

    def forward(self, src):
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.linear_out(output[-1])
        return output

# Initialize the Transformer model
model_transformer = TransformerPredictor(d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=256, dropout=0.1)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_transformer.to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model_transformer.parameters(), lr=0.001)

def train_transformer(inputs, targets):
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).to(device)
    
    model_transformer.train()
    optimizer.zero_grad()
    outputs = model_transformer(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict_transformer(inputs):
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1).to(device)
    model_transformer.eval()
    with torch.no_grad():
        outputs = model_transformer(inputs)
    return outputs.cpu().numpy()[0]

# Display function
def draw(frame, detected_targets, locked_target, zoom_center):
    screen.fill(WHITE)
    
    # Draw detected targets
    for target in detected_targets:
        center_x, center_y = target[0] + target[2] // 2, target[1] + target[3] // 2
        pygame.draw.circle(screen, RED, (center_x, center_y), 5)  # Adjust circle size for visibility
    
    # Draw locked target
    if locked_target:
        pygame.draw.circle(screen, GREEN, locked_target, 5)
    
    # Display zoomed-in view if a target is locked
    if zoom_center:
        sub_frame = frame[max(zoom_center[1] - height // 2, 0):zoom_center[1] + height // 2,
                          max(zoom_center[0] - width // 2, 0):zoom_center[0] + width // 2]
        sub_frame_resized = cv2.resize(sub_frame, (width, height))
        pygame.surfarray.blit_array(screen, cv2.cvtColor(sub_frame_resized, cv2.COLOR_BGR2RGB))
    
    # Display status
    font = pygame.font.Font(None, 36)
    status_text = f"Locked Target: {locked_target} | Zoom Center: {zoom_center}"
    text_surface = font.render(status_text, True, BLUE)
    screen.blit(text_surface, (10, 10))
    
    pygame.display.flip()

# Logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Main loop
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                # Left-click to lock target
                if event.button == 1:
                    closest_target = None
                    min_distance = float('inf')
                    for target in detected_targets:
                        center_x, center_y = target[0] + target[2] // 2, target[1] + target[3] // 2
                        distance = (mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2
                        if distance < min_distance:
                            closest_target = (center_x, center_y)
                            min_distance = distance

                    # Check if the closest target is within 3mm threshold
                    if closest_target and np.sqrt(min_distance) <= target_radius:
                        locked_target = closest_target
                        x_history.append(locked_target[0])
                        y_history.append(locked_target[1])
                        
                        # Train the Transformer model with the history data
                        if len(x_history) == history_length:
                            inputs = list(zip(x_history, y_history))
                            targets = [(x_history[-1], y_history[-1])]
                            loss = train_transformer(inputs, targets)
                            logging.info(f"Training loss: {loss}")

                        zoom_center = locked_target

                # Right-click to unlock target and reset zoom
                elif event.button == 3:
                    locked_target = None
                    zoom_center = None

        ret, frame = cap.read()
        if not ret:
            raise IOError("Failed to capture frame")

        # Detect objects
        detected_targets = detect_objects(frame)

        # Draw the current state
        draw(frame, detected_targets, locked_target, zoom_center)

except KeyboardInterrupt:
    logging.info("Application terminated by user")
except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    cap.release()
    pygame.quit()
    sys.exit()
