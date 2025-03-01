import pygame
import cv2
import sys
import os
import subprocess
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from collections import deque

# Function to check and install required libraries
def ensure_installed(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ["pygame", "opencv-python-headless", "numpy", "tensorflow"]
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
try:
    import tensorflow as tf
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format

    # Path to the pipeline config and checkpoint directory
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

except ImportError:
    print("TensorFlow or Object Detection API is not installed. Please ensure they are installed.")
    sys.exit(1)

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

# Zoom parameters
zoom_factor = 2.0
zoom_center = None

def draw():
    screen.fill(WHITE)

    # Draw detected targets
    for target in detected_targets:
        pygame.draw.circle(screen, RED, target, target_radius)

    # Highlight closest target to mouse pointer if not locked
    if not locked_target and detected_targets:
        closest_distance = float('inf')
        closest_target = None
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for target in detected_targets:
            distance = (mouse_x - target[0]) ** 2 + (mouse_y - target[1]) ** 2
            if distance < closest_distance:
                closest_distance = distance
                closest_target = target

        # Highlight the closest target
        if closest_target:
            pygame.draw.circle(screen, GREEN, closest_target, target_radius)

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
        sub_surface = screen.subsurface((x - width // 4, y - height // 4, width // 2, height // 2))
        sub_surface_scaled = pygame.transform.scale(sub_surface, (width, height))
        screen.blit(sub_surface_scaled, (0, 0))

    pygame.display.flip()

# Main loop
running = True
detected_targets = []
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            # Check if the mouse click is within any detected target circle
            for target in detected_targets:
                distance = (mouse_x - target[0]) ** 2 + (mouse_y - target[1]) ** 2
                if distance <= target_radius ** 2:
                    locked_target = target
                    x_history.append(locked_target[0])
                    y_history.append(locked_target[1])
                    update_predictive_model()
                    break

            # Right mouse button or number '3' key for zooming
            if event.button == 3:
                if locked_target:
                    zoom_center = locked_target

    # Handle keyboard input for adaptive behavior and zooming
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        target_x -= 5
    if keys[pygame.K_RIGHT]:
        target_x += 5
    if keys[pygame.K_UP]:
        target_y -= 5
    if keys[pygame.K_DOWN]:
        target_y += 5
    if keys[pygame.K_3]:  # Number 3 key for zooming
        if locked_target:
            zoom_center = locked_target

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

    draw()

# Clean up
cap.release()
pygame.quit()
sys.exit()
