def calculate_risk_score(density_score, motion_score, pose_score, num_people):
    """
    Returns a risk level: Low, Medium, High
    """
    if num_people < 3:
        return "Low"

    if density_score > 0.5 or motion_score > 0.4 or pose_score:
        return "High"
    elif density_score > 0.3 or motion_score > 0.2:
        return "Medium"
    else:
        return "Low"