import pygame
import cv2
import sys
import os
import subprocess
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from collections import deque
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Function to check and install required libraries
def ensure_installed(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ["pygame", "opencv-python-headless", "numpy", "scikit-learn", "tensorflow", "keras"]
for package in required_packages:
    ensure_installed(package)

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Advanced Mouse Tracking and Targeting")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Load pre-trained model for object detection
pipeline_config = os.path.join('models', 'research', 'object_detection', 'samples', 'configs', 'faster_rcnn_inception_v2_pets.config')
checkpoint_dir = os.path.join('models', 'research', 'object_detection', 'model_zoo', 'faster_rcnn_inception_v2_coco_2018_01_28')

# Load the pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
detection_model = tf.saved_model.load(checkpoint_dir)

def detect_objects(image, detection_model):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    detections = detection_model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
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
    model = load_model(model_path)
else:
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

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            # Check if the mouse click is within any detected target circle
            if locked_target and (mouse_x - locked_target[0]) ** 2 + (mouse_y - locked_target[1]) ** 2 <= target_radius ** 2:
                if event.button == 1:  # Left mouse button
                    x_history.append(mouse_x)
                    y_history.append(mouse_y)
                    update_predictive_model()
                elif event.button == 3:  # Right mouse button
                    locked_target = None

    # Get webcam frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB and resize for object detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detect_objects(frame_rgb, detection_model)

    # Clear detected targets
    detected_targets = []

    # Draw detected objects
    for i in range(detections['num_detections']):
        if detections['detection_scores'][i] > 0.5:
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
            im_height, im_width, _ = frame.shape
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            center_x = int((left + right) / 2)
            center_y = int((top + bottom) / 2)

            # Draw bounding box and circle
            cv2.rectangle(frame_rgb, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            pygame.draw.circle(screen, RED, (center_x, center_y), target_radius)

            detected_targets.append((center_x, center_y))

    # Auto-select nearest target
    if not locked_target and detected_targets:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        closest_distance = float('inf')
        closest_target = None
        for target in detected_targets:
            distance = (mouse_x - target[0]) ** 2 + (mouse_y - target[1]) ** 2
            if distance < closest_distance:
                closest_distance = distance
                closest_target = target

        locked_target = closest_target
        x_history.append(locked_target[0])
        y_history.append(locked_target[1])
        update_predictive_model()

    # Predictive tracking
    if locked_target is not None:
        predicted_target = predict_target()
        if predicted_target is not None:
            pygame.draw.circle(screen, GREEN, predicted_target, target_radius)

    draw()

    # Handle keyboard input for adaptive behavior
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        target_x -= 5
    if keys[pygame.K_RIGHT]:
        target_x += 5
    if keys[pygame.K_UP]:
        target_y -= 5
    if keys[pygame.K_DOWN]:
        target_y += 5

    # Display the webcam feed on the left side of the screen
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    screen.blit(pygame.transform.scale(frame_surface, (width // 2, height)), (0, 0))

cap.release()
pygame.quit()
sys.exit()
