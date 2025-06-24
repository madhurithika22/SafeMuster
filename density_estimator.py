# ai_models/density_estimator.py

import cv2
import numpy as np

def generate_density_heatmap(frame, boxes):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(heatmap, (cx, cy), radius=15, color=1.0, thickness=-1)

    density_score = np.clip(np.sum(heatmap) / 500, 0.0, 1.0)

    if np.max(heatmap) > 0:
        heatmap_colored = cv2.applyColorMap((255 * heatmap / np.max(heatmap)).astype(np.uint8), cv2.COLORMAP_JET)
        frame_overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
    else:
        frame_overlay = frame.copy()

    return frame_overlay, round(density_score, 2)