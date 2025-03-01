import cv2
import numpy as np
import time

# Load the pre-trained model (YOLO in this case)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam
cap = cv2.VideoCapture(0)

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Post-process the outputs
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)

            # Check if the locked target is within this box
            if locked_target == label:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)  # Highlight locked target

    # Display the frame
    cv2.imshow("Target Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
