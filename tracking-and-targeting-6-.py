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

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Positional Encoding
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

# TransformerPredictor
class TransformerPredictor(torch.nn.Module):
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
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
model_transformer = TransformerPredictor(d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=256, dropout=0.1).to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model_transformer.parameters(), lr=0.001)

# Training loop with early stopping
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

# Prediction
def predict_transformer(inputs):
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1).to(device)
    model_transformer.eval()
    with torch.no_grad():
        outputs = model_transformer(inputs)
    return outputs.cpu().numpy()[0]

# Mock Evaluation Function
def mock_evaluation(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss

# Data Augmentation
def augment_data(data):
    augmented_data = []
    for x, y in data:
        # Add noise
        noisy_x = x + np.random.normal(0, 1, size=x.shape)
        noisy_y = y + np.random.normal(0, 1, size=y.shape)
        augmented_data.append((noisy_x, noisy_y))
        
        # Random scaling
        scale_factor = np.random.uniform(0.8, 1.2)
        scaled_x = x * scale_factor
        scaled_y = y * scale_factor
        augmented_data.append((scaled_x, scaled_y))
    
    return augmented_data

# Display function
def draw(frame, detected_targets, locked_target, zoom_center, predicted_path):
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
    
    # Draw predicted path
    if predicted_path:
        for i in range(len(predicted_path) - 1):
            pygame.draw.line(screen, BLUE, predicted_path[i], predicted_path[i+1], 2)
    
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
    # Initialize variables
    history_length = 5
    x_history = deque(maxlen=history_length)
    y_history = deque(maxlen=history_length)
    locked_target = None
    zoom_center = None
    
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

                    # Check if the closest target is within 3mm threshold (0.12 pixels at 800x600 resolution)
                    if closest_target and np.sqrt(min_distance) <= 0.12:
                        locked_target = closest_target
                        x_history.append(locked_target[0])
                        y_history.append(locked_target[1])
                        
                        # Train the Transformer model with the history data
                        if len(x_history) == history_length:
                            inputs = list(zip(x_history, y_history))
                            targets = [(x_history[-1], y_history[-1])]
                            
                            # Augment data
                            augmented_data = augment_data([(inputs, targets)])
                            for aug_inputs, aug_targets in augmented_data:
                                loss = train_transformer(aug_inputs, aug_targets)
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

        # Predict path
        predicted_path = []
        if len(x_history) == history_length:
            inputs = list(zip(x_history, y_history))
            prediction = predict_transformer(inputs)
            predicted_path.append((int(prediction[0]), int(prediction[1])))

        # Draw the current state
        draw(frame, detected_targets, locked_target, zoom_center, predicted_path)

except KeyboardInterrupt:
    logging.info("Application terminated by user")
except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    cap.release()
    pygame.quit()
    sys.exit()

# Mock data loader for evaluation
def mock_data_loader(num_samples=100, history_length=5):
    data = []
    for _ in range(num_samples):
        x_history = np.random.rand(history_length)
        y_history = np.random.rand(history_length)
        target_x = np.random.rand()
        target_y = np.random.rand()
        inputs = list(zip(x_history, y_history))
        targets = [(target_x, target_y)]
        data.append((inputs, targets))
    return data

# Create mock dataset
mock_dataset = mock_data_loader(num_samples=1000, history_length=5)

# Train the model with mock data
for epoch in range(10):
    total_loss = 0.0
    for inputs, targets in mock_dataset:
        loss = train_transformer(inputs, targets)
        total_loss += loss
    avg_loss = total_loss / len(mock_dataset)
    logging.info(f"Epoch {epoch+1}, Training Loss: {avg_loss}")

# Evaluate the model with mock data
evaluation_loss = mock_evaluation(model_transformer, mock_dataset, criterion)
logging.info(f"Evaluation Loss: {evaluation_loss}")
