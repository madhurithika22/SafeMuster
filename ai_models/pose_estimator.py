import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def detect_pose_fall(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return False

    landmarks = results.pose_landmarks.landmark

    # Get key landmark positions (in normalized coords)
    try:
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    except:
        return False

    # Calculate average y positions
    hip_y = (left_hip.y + right_hip.y) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    nose_y = nose.y

    # Heuristic: if hips and nose are almost at same level (horizontal fall)
    # and shoulders are low in y-axis too, it may be a fall
    vertical_gap = abs(nose_y - hip_y)
    shoulder_hip_gap = abs(shoulder_y - hip_y)

    if vertical_gap < 0.05 and shoulder_hip_gap < 0.05:
        return True  # possible fall

    return False
