import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from models.experimental import attempt_load  # YOLOv5
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer
import cv2
import os

# Auto-load necessary libraries
try:
    import torchvision
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError:
    os.system('pip install torch torchvision pillow matplotlib')
    import torchvision
    from PIL import Image
    import matplotlib.pyplot as plt

# Load pre-trained models
def load_models():
    ssd_model = ssd300_vgg16(pretrained=True)
    ssd_model.eval()

    yolo_model = attempt_load('yolov5s.pt', map_location='cpu')
    yolo_model.eval()

    faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
    faster_rcnn_model.eval()

    mask_rcnn_model = maskrcnn_resnet50_fpn(pretrained=True)
    mask_rcnn_model.eval()

    return ssd_model, yolo_model, faster_rcnn_model, mask_rcnn_model

# Transform image to tensor
def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    return img_tensor

# Draw bounding boxes on frame
def draw_bounding_boxes(frame, detections):
    for det in detections:
        box = det['box']
        label = det['label']
        score = det['score']
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {score:.2f}', (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Compute Average Precision
def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# Evaluate detections
def evaluate_detections(ground_truth, predictions):
    all_boxes = []
    all_scores = []
    all_labels = []
    all_true_boxes = []
    all_true_labels = []

    for gt, pred in zip(ground_truth, predictions):
        true_boxes = np.array(gt['boxes'])
        true_labels = np.array(gt['labels'])

        pred_boxes = np.array([det['box'] for det in pred])
        pred_scores = np.array([det['score'] for det in pred])
        pred_labels = np.array([det['label'] for det in pred])

        all_true_boxes.append(true_boxes)
        all_true_labels.append(true_labels)

        all_boxes.append(pred_boxes)
        all_scores.append(pred_scores)
        all_labels.append(pred_labels)

    all_true_boxes = np.concatenate(all_true_boxes)
    all_true_labels = np.concatenate(all_true_labels)

    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    unique_labels = np.unique(np.concatenate((all_true_labels, all_labels)))

    aps = []
    for label in unique_labels:
        true_indices = all_true_labels == label
        pred_indices = all_labels == label

        true_boxes_label = all_true_boxes[true_indices]
        pred_boxes_label = all_boxes[pred_indices]
        pred_scores_label = all_scores[pred_indices]

        if len(true_boxes_label) == 0 or len(pred_boxes_label) == 0:
            aps.append(0)
            continue

        # Sort predictions by score in descending order
        sorted_indices = np.argsort(-pred_scores_label)
        pred_boxes_label = pred_boxes_label[sorted_indices]
        pred_scores_label = pred_scores_label[sorted_indices]

        tp = np.zeros(len(pred_scores_label))
        fp = np.zeros(len(pred_scores_label))

        for i, box in enumerate(pred_boxes_label):
            overlaps = np.array([iou(box, tbox) for tbox in true_boxes_label])
            max_overlap_index = np.argmax(overlaps)
            if overlaps[max_overlap_index] >= 0.5:
                tp[i] = 1
                true_boxes_label = np.delete(true_boxes_label, max_overlap_index, axis=0)
            else:
                fp[i] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp / len(true_boxes_label) if len(true_boxes_label) > 0 else np.zeros_like(tp)
        prec = tp / (tp + fp)

        aps.append(compute_ap(rec, prec))

    mAP = np.mean(aps)

    return {
        'mAP': mAP,
        'precision': precision_score(all_true_labels, all_labels, average='weighted'),
        'recall': recall_score(all_true_labels, all_labels, average='weighted'),
        'f1_score': f1_score(all_true_labels, all_labels, average='weighted')
    }

# Intersection over Union (IoU)
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou_score = interArea / float(boxAArea + boxBArea - interArea)
    return iou_score

# Perform real-time detection
def perform_detection(models, input_source):
    ssd_model, yolo_model, faster_rcnn_model, mask_rcnn_model = models

    if input_source.endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(input_source)
    else:
        frame = cv2.imread(input_source)
        cap = None

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
        img_tensor = preprocess_image(frame)

        # Perform detection using SSD
        ssd_detections = ssd_model(img_tensor)
        ssd_results = [{'box': det['boxes'][i].tolist(), 'label': det['labels'][i].item(), 'score': det['scores'][i].item()}
                       for i in range(len(det['boxes'])) if det['scores'][i] > 0.5]

        # Perform detection using YOLO
        yolo_results = yolo_model(img_tensor)
        yolo_results = [{'box': det[2:6].tolist(), 'label': det[1].item(), 'score': det[4].item()}
                        for det in yolo_results.xyxy[0] if det[4] > 0.5]

        # Perform detection using Faster R-CNN
        faster_rcnn_detections = faster_rcnn_model(img_tensor)
        faster_rcnn_results = [{'box': det['boxes'][i].tolist(), 'label': det['labels'][i].item(), 'score': det['scores'][i].item()}
                               for i in range(len(det['boxes'])) if det['scores'][i] > 0.5]

        # Perform detection using Mask R-CNN
        mask_rcnn_detections = mask_rcnn_model(img_tensor)
        mask_rcnn_results = [{'box': det['boxes'][i].tolist(), 'label': det['labels'][i].item(), 'score': det['scores'][i].item()}
                             for i in range(len(det['boxes'])) if det['scores'][i] > 0.5]

        # Combine results
        all_results = ssd_results + yolo_results + faster_rcnn_results + mask_rcnn_results

        # Draw bounding boxes on frame
        annotated_frame = draw_bounding_boxes(frame, all_results)

        # Display the frame
        cv2.imshow('Annotated Frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Load models
    models = load_models()

    # Define ground truth and predictions for evaluation
    ground_truth_data = [
        {'image_id': 'img1', 'boxes': [[10, 20, 30, 40], [50, 60, 70, 80]], 'labels': [1, 2]},
        # Add more ground truth data as needed
    ]

    predicted_data = [
        {'image_id': 'img1', 'boxes': [[9, 21, 31, 41], [51, 61, 71, 81]], 'scores': [0.9, 0.8], 'labels': [1, 2]},
        # Add more predicted data as needed
    ]

    # Evaluate detection performance
    evaluation_metrics = evaluate_detections(ground_truth_data, predicted_data)
    print("Evaluation Metrics:", evaluation_metrics)

    # Perform real-time detection on a video or image
    input_source = 'video.mp4'  # Change to your video file path or an image file path
    perform_detection(models, input_source)

if __name__ == "__main__":
    main()
