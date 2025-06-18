# ai_models/risk_scorer.py

def calculate_risk_score(density_score, motion_score, fall_detected):
    score = 0.0
    if density_score > 0.6:
        score += 0.5
    if motion_score > 0.5:
        score += 0.3
    if fall_detected:
        score += 0.2

    if score > 0.6:
        level = "HIGH"
    elif score > 0.3:
        level = "MEDIUM"
    else:
        level = "LOW"

    return round(score, 2), level