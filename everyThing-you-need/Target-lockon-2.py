import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from imgaug.augmenters import Sequential, SomeOf, Multiply, AddToHueAndSaturation, GaussianBlur, PepperAndSalt
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast

# Check if required libraries are installed
try:
    import cv2
    import numpy as np
    import time
    import torch
    from ultralytics import YOLO
    from imgaug.augmenters import Sequential, SomeOf, Multiply, AddToHueAndSaturation, GaussianBlur, PepperAndSalt
    from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast
except ImportError as e:
    print(f"Missing library: {e}")
    print("Please install the required libraries using:")
    print("pip install opencv-python-headless onnxruntime ultralytics imgaug albumentations numpy")
    exit(1)

# Load the pre-trained YOLOv8 model
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    print("Ensure that 'yolov8n.pt' is in the current directory or provide the correct path.")
    exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Variables to track mouse events
mouse_down_time = None
mouse_up_time = None
locked_target = None

def mouse_callback(event, x, y, flags, param):
    global mouse_down_time, mouse_up_time, locked_target
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down_time = time.time()
    
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_up_time = time.time()
        click_duration = mouse_up_time - mouse_down_time
        
        if click_duration < 0.09 and locked_target is not None:
            print(f"Locked onto target: {locked_target}")
        else:
            locked_target = None

# Set the mouse callback function
cv2.namedWindow("Target Tracking")
cv2.setMouseCallback("Target Tracking", mouse_callback)

# Load class names
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: 'coco.names' file not found.")
    exit(1)
except Exception as e:
    print(f"Error reading 'coco.names': {e}")
    exit(1)

# Data augmentation pipeline using imgaug and albumentations
def augment_image(image):
    seq = Sequential([
        SomeOf((0, 2), [
            Multiply((0.75, 1.25)),
            AddToHueAndSaturation((-10, 10)),
            GaussianBlur((0, 3.0)),
            PepperAndSalt(0.05)
        ])
    ], random_order=True)

    # Apply imgaug augmentations
    image_aug = seq(image=image)["image"]

    # Apply albumentations augmentations
    transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
    ])
    
    image_aug = transform(image=image_aug)["image"]

    return image_aug

# Adaptive thresholding function
def adaptive_threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    return thresh

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Apply adaptive thresholding
        thresh = adaptive_threshold(frame)

        # Perform object detection using YOLOv8
        results = model(frame, conf=0.5)  # Confidence threshold

        boxes = []
        confidences = []
        class_ids = []

        for result in results:
            for box in result.boxes.cpu().numpy():
                r = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls = int(box.cls[0])

                if conf > 0.5:  # Confidence threshold
                    boxes.append(r)
                    confidences.append(conf)
                    class_ids.append(cls)

        # Non-maximum suppression (NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x1, y1, x2, y2 = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                
                # Highlight locked target
                if locked_target == label:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Red rectangle for locked target
                    color = (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)

        # Display the frame
        cv2.imshow("Target Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
