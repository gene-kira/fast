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

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

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
target_radius = 15
locked_target = None

# LSTM model for predictive tracking
history_length = 10
x_history = deque(maxlen=history_length)
y_history = deque(maxlen=history_length)

# Build or load LSTM model
model_path = 'lstm_model.h5'
if os.path.exists(model_path):
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, input_shape=(history_length, 2), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')

def update_predictive_model():
    if len(x_history) == history_length and len(y_history) == history_length:
        X = np.array([list(zip(x_history, y_history))])
        y = np.array([[x_history[-1], y_history[-1]]])
        model.fit(X, y, epochs=5, verbose=0)

def predict_target():
    if len(x_history) == history_length and len(y_history) == history_length:
        X = np.array([list(zip(x_history, y_history))])
        prediction = model.predict(X)
        return (int(prediction[0][0]), int(prediction[0][1]))
    return None

# Zoom parameters
zoom_factor = 2.0
zoom_center = None

def draw():
    screen.fill(WHITE)

    # Draw detected targets
    for target in detected_targets:
        center_x, center_y = target[0] + target[2] // 2, target[1] + target[3] // 2
        pygame.draw.circle(screen, RED, (center_x, center_y), target_radius)
        pygame.draw.rect(screen, RED, target)

    # Highlight closest target to mouse pointer if not locked
    if not locked_target and detected_targets:
        closest_distance = float('inf')
        closest_target = None
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for target in detected_targets:
            center_x, center_y = target[0] + target[2] // 2, target[1] + target[3] // 2
            distance = (mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2
            if distance < closest_distance:
                closest_distance = distance
                closest_target = target

        # Highlight the closest target
        if closest_target:
            center_x, center_y = closest_target[0] + closest_target[2] // 2, closest_target[1] + closest_target[3] // 2
            pygame.draw.circle(screen, GREEN, (center_x, center_y), target_radius)
            # Snap to closest target if within a certain threshold
            if closest_distance <= (target_radius * 1.5) ** 2:
                locked_target = (center_x, center_y)
                x_history.append(locked_target[0])
                y_history.append(locked_target[1])
                update_predictive_model()
                zoom_center = locked_target

    # Draw locked target
    if locked_target:
        pygame.draw.circle(screen, GREEN, locked_target, target_radius)

    # Predictive tracking
    if locked_target is not None and zoom_center is None:
        predicted_target = predict_target()
        if predicted_target is not None:
            pygame.draw.circle(screen, GREEN, predicted_target, target_radius)

    # Zoom in on the locked target
    if zoom_center:
        x, y = zoom_center
        sub_frame = frame[max(y - height // 2, 0):y + height // 2, max(x - width // 2, 0):x + width // 2]
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
                    for target in detected_targets:
                        center_x, center_y = target[0] + target[2] // 2, target[1] + target[3] // 2
                        distance = (mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2
                        if distance <= target_radius ** 2:
                            locked_target = (center_x, center_y)
                            x_history.append(locked_target[0])
                            y_history.append(locked_target[1])
                            update_predictive_model()
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

        draw()

except KeyboardInterrupt:
    logging.info("Application terminated by user")
except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    cap.release()
    pygame.quit()
    sys.exit()
