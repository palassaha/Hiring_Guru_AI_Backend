
import json
import os
from app.interview.emotion_detector import score_nervousness_relative

def analyze_nervousness(features: dict) -> dict:
    if not os.path.exists("baseline_features.json"):
        print("[⚠️] No baseline features found. Skipping comparison.")
        return {"nervousness_score": 0.0, "label": "Unknown"}

    with open("baseline_features.json") as f:
        baseline = json.load(f)

    relative = {}
    for key in features:
        if key in baseline:
            base = baseline[key]
            if base != 0:
                relative[key] = (features[key] - base) / abs(base)
            else:
                relative[key] = 0.0

    print(f"[📉] Relative Feature Changes: {relative}")
    return score_nervousness_relative(relative)
