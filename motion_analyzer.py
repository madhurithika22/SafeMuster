# ai_models/motion_analyzer.py

import cv2
import numpy as np

def calculate_motion_score(prev_frame, current_frame):
    if prev_frame is None or current_frame is None:
        return 0.0

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    score = np.sum(diff) / 100000.0
    return round(score, 2)