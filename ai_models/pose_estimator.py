# ai_models/pose_estimator.py

import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def detect_pose_fall(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        left_hip = results.pose_landmarks.landmark[23]
        right_hip = results.pose_landmarks.landmark[24]
        hip_y = (left_hip.y + right_hip.y) / 2
        if nose.y > hip_y:  # head lower than hips
            return True
    return False