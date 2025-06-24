from ultralytics import YOLO
import cv2

# Load YOLOv8 model (make sure yolov8n.pt or yolov8s.pt is downloaded)
model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

def detect_people(frame):
    # Resize frame to improve detection accuracy (optional but recommended)
    resized_frame = cv2.resize(frame, (1280, 720))

    # Perform inference (set lower confidence threshold to catch smaller people)
    results = model.predict(resized_frame, conf=0.1, classes=[0])  # class 0 = person

    person_boxes = []

    for result in results:
        for box in result.boxes:
            # Filter only 'person' class
            cls_id = int(box.cls[0])
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))

    return person_boxes