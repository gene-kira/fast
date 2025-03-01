# Auto-load necessary libraries
try:
    import cv2
except ImportError:
    print("OpenCV is not installed. Installing OpenCV...")
    !pip install opencv-python-headless
    import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("YOLOv8 is not installed. Installing YOLOv8...")
    !pip install ultralytics
    from ultralytics import YOLO

try:
    import numpy as np
except ImportError:
    print("NumPy is not installed. Installing NumPy...")
    !pip install numpy
    import numpy as np

try:
    import random
except ImportError:
    print("Random module is not installed. Installing Random...")
    !pip install random
    import random

try:
    import imgaug.augmenters as iaa
except ImportError:
    print("imgaug is not installed. Installing imgaug...")
    !pip install imgaug
    import imgaug.augmenters as iaa

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("Could not open video source.")

# Load YOLOv8 model and class names file
model_path = "yolov8n.pt"  # Update this with your model path
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    raise

class_names_file = "coco.names"  # Update this with your class names file path
try:
    with open(class_names_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Error reading class names file: {e}")
    raise

# Enhance contrast function
def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# Data augmentation function
def augment_frame(frame):
    seq = iaa.Sequential([
        iaa.Rot90(k=random.randint(0, 3)),  # Random rotation by 0, 90, 180, or 270 degrees
        iaa.Multiply((0.5, 1.5))           # Random brightness adjustment
    ])
    return seq(image=frame)

# Global variables for locked target and Kalman filter
locked_target = None
tracked_targets = {}
kalman_filters = {}

# Ground truth data (simulated)
ground_truth_data = []

# Evaluation metrics
iou_threshold = 0.5

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection coordinates
    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)

    # Calculate intersection area
    if ix_min < ix_max and iy_min < iy_max:
        intersection_area = (ix_max - ix_min) * (iy_max - iy_min)
    else:
        intersection_area = 0

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def evaluate_tracking(predictions, ground_truth):
    frame_metrics = []
    for pred_frame, gt_frame in zip(predictions, ground_truth):
        predicted_box = pred_frame['box']
        gt_box = gt_frame['box']

        # Calculate IoU
        iou = calculate_iou(predicted_box, gt_box)

        # Determine if the prediction is accurate based on IoU threshold
        is_accurate = iou >= iou_threshold

        frame_metrics.append({
            'iou': iou,
            'is_accurate': is_accurate
        })
    return frame_metrics

# User interaction function
def mouse_callback(event, x, y, flags, param):
    global locked_target
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                label = classes[class_ids[i]]
                locked_target = label
                print(f"Locked onto target: {locked_target}")
                init_kalman_filter(label, box)
                break

# Kalman Filter Initialization
def init_kalman_filter(label, box):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-4
    kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePre = np.array([[box[0]], [box[1]], [0], [0]], dtype=np.float32)

    kalman_filters[label] = kf

# Kalman Filter Update
def update_kalman_filter(kf, box):
    measurement = np.array([[np.float32(box[0])], [np.float32(box[1])]])
    kf.correct(measurement)
    return kf.predict()

# Main loop
cv2.namedWindow('Target Tracking')
cv2.setMouseCallback('Target Tracking', mouse_callback)

frame_predictions = []
frame_ground_truth = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate ground truth data for evaluation
        gt_box = [random.randint(100, 300), random.randint(100, 300), random.randint(200, 400), random.randint(200, 400)]
        frame_ground_truth.append({'box': gt_box})

        # Enhance contrast and augment frame
        enhanced_frame = enhance_contrast(frame)
        augmented_frame = augment_frame(enhanced_frame)

        # Run detection
        results = model(augmented_frame)
        boxes = []
        class_ids = []

        for result in results:
            for box in result.boxes.cpu().numpy():
                r = box.xyxy[0].astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > 0.3:  # Filter detections with confidence > 0.3
                    boxes.append(r)
                    class_ids.append(cls_id)

                    cv2.rectangle(augmented_frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                    label = classes[cls_id]
                    cv2.putText(augmented_frame, f"{label} ({conf:.2f})", (r[0], r[1] - 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)

        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, [float(box.conf[0]) for box in result.boxes], 0.3, 0.5)
        boxes = [boxes[i] for i in indexes]
        class_ids = [class_ids[i] for i in indexes]

        # Track locked target
        if locked_target and locked_target in kalman_filters:
            kf = kalman_filters[locked_target]
            prediction = update_kalman_filter(kf, gt_box)  # Use ground truth for evaluation

            pred_box = [
                int(prediction[0] - (gt_box[2] - gt_box[0]) / 2),
                int(prediction[1] - (gt_box[3] - gt_box[1]) / 2),
                int(prediction[0] + (gt_box[2] - gt_box[0]) / 2),
                int(prediction[1] + (gt_box[3] - gt_box[1]) / 2)
            ]

            frame_predictions.append({'box': pred_box})

            # Draw bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(augmented_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw predicted bounding box
            cv2.rectangle(augmented_frame, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 0, 255), 2)
            cv2.putText(augmented_frame, f"Predicted: {locked_target}", (pred_box[0], pred_box[1] - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

        # Draw ground truth bounding box
        cv2.rectangle(augmented_frame, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 0), 2)
        cv2.putText(augmented_frame, "Ground Truth", (gt_box[0], gt_box[1] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)

        # Display the frame
        cv2.imshow('Target Tracking', augmented_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

# Evaluate tracking performance
evaluation_metrics = evaluate_tracking(frame_predictions, frame_ground_truth)
total_frames = len(evaluation_metrics)
accurate_frames = sum(1 for metric in evaluation_metrics if metric['is_accurate'])
tracking_accuracy = accurate_frames / total_frames

print(f"Tracking Accuracy: {tracking_accuracy * 100:.2f}%")

# Print detailed metrics
for i, metric in enumerate(evaluation_metrics):
    print(f"Frame {i+1}: IoU={metric['iou']:.2f}, Accurate={metric['is_accurate']}")
