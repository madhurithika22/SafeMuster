import os
import sys

# Add current directory to sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import numpy as np
from ultralytics import YOLO

# AI model imports
from ai_models.density_estimator import generate_density_heatmap
from ai_models.motion_analyzer import calculate_motion_score
from ai_models.pose_estimator import detect_pose_fall
from ai_models.risk_scorer import calculate_risk_score
from ai_models.person_detector import detect_people  # even if unused
from alert import send_alert


def run_pipeline(video_source):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
   
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    prev_frame = None
    alert_triggered = False

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
                person_boxes = [box[:4] for box in data if int(box[5]) == 0]

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
        risk_level = calculate_risk_score(
            density_score=density_score,
            motion_score=motion_score,
            pose_score=fall_detected,
            num_people=num_people
        )

        # ---- Trigger Alert If Needed ----
        if risk_level == "High":
            if not alert_triggered:
                print("🚨 High Risk Detected – Triggering Alert!")
                send_alert()
                alert_triggered = True
        else:
            alert_triggered = False

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
            cv2.putText(display_frame, f"Risk Level: {risk_level}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if risk_level == "High" else
                        (0, 165, 255) if risk_level == "Medium" else
                        (0, 255, 0), 2)
            cv2.putText(display_frame, f"Risk Alert: {'YES' if risk_level == 'High' else 'NO'}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255) if risk_level == "High" else (0, 255, 0), 3)
        except cv2.error as e:
            print("OpenCV Error:", e)

        # ---- Yield frame to Flask ----
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield frame_bytes, risk_level, alert_triggered

    cap.release()

