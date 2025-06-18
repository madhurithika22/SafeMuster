import cv2
import sys
import numpy as np
from ultralytics import YOLO
from ai_models.density_estimator import generate_density_heatmap
from ai_models.pose_estimator import detect_pose_fall
from ai_models.motion_analyzer import calculate_motion_score
from ai_models.risk_scorer import calculate_risk_score

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Get video source from CLI (default: webcam)
video_source = 0  # default webcam
if len(sys.argv) > 1:
    video_source = sys.argv[1]

# Initialize video capture
cap = cv2.VideoCapture(video_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"Error: Could not open video source: {video_source}")
    sys.exit(1)

prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error.")
        break

    # ---- Person Detection ----
    results = model(frame)
    person_boxes = []
    
    if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
        boxes = results[0].boxes
        if hasattr(boxes, "data"):
            data = boxes.data.cpu().numpy()
            person_boxes = [box[:4] for box in data if int(box[5]) == 0]  # class 0 = person

    num_people = len(person_boxes)

    # ---- Density Estimation ----
    frame_with_heatmap, density_score = generate_density_heatmap(frame, person_boxes)

    # ---- Motion Detection ----
    motion_score = 0.0
    if prev_frame is not None:
        motion_score = calculate_motion_score(prev_frame, frame)
    prev_frame = frame.copy()

    # ---- Pose Estimation ----
    fall_detected = detect_pose_fall(frame) if density_score > 0.4 else False

    # ---- Risk Calculation ----
    risk_score, risk_condition = calculate_risk_score(
        density_score=density_score,
        motion_score=motion_score,
        fall_detected=fall_detected
    )

    # ---- Overlay Display ----
    display_frame = frame_with_heatmap.copy()

    try:
        cv2.putText(display_frame, f"People: {num_people}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Density: {density_score:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Motion: {motion_score:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Fall Detected: {fall_detected}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if fall_detected else (0, 0, 255), 2)
        cv2.putText(display_frame, f"Risk Score: {risk_score:.2f}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        cv2.putText(display_frame, f"Risk Alert: {'YES' if risk_condition else 'NO'}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if risk_condition else (0, 255, 0), 3)
    except cv2.error as e:
        print("OpenCV Error in putText:", e)

    cv2.imshow("SafeMuster - Stampede Prediction", display_frame)

    key = cv2.waitKey(1 if video_source == "0" or video_source == 0 else 25)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()