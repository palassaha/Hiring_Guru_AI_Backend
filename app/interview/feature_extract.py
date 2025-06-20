
import os
import json
import parselmouth
from parselmouth.praat import call

def extract_voice_features(audio_path: str) -> dict:
    snd = parselmouth.Sound(audio_path)
    
    pitch = snd.to_pitch()
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    features = {
        "mean_pitch_hz": call(pitch, "Get mean", 0, 0, "Hertz"),
        "jitter_local": call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
        "shimmer_local": call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        "hnr": call(harmonicity, "Get mean", 0, 0)
    }

    return features


def compute_relative_features(current: dict, baseline: dict) -> dict:
    """
    Computes relative deltas from the baseline features.
    Returns a dictionary with change ratios.
    """
    deltas = {}
    for key in current:
        base = baseline.get(key, 1e-6)  # avoid divide-by-zero
        delta = (current[key] - base) / base
        deltas[key] = round(delta, 4)
    return deltas


def load_baseline(path="baseline_features.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError("Baseline features not found.")
    with open(path, "r") as f:
        return json.load(f)
