
def score_nervousness_relative(relative_features: dict) -> dict:
    """
    Estimate nervousness based on relative increase from baseline.
    Each delta is a percentage change from the baseline.
    """

    jitter_delta = relative_features.get("jitter_local", 0)
    shimmer_delta = relative_features.get("shimmer_local", 0)
    hnr_delta = -relative_features.get("hnr", 0)  # Decrease in HNR is bad
    pitch_delta = relative_features.get("mean_pitch_hz", 0)

    # Clamp deltas
    jitter_score = min(max(jitter_delta, 0), 1.0)
    shimmer_score = min(max(shimmer_delta, 0), 1.0)
    hnr_score = min(max(hnr_delta, 0), 1.0)
    pitch_score = min(max(abs(pitch_delta), 0), 1.0)

    nervousness_score = (
        0.3 * jitter_score +
        0.3 * shimmer_score +
        0.3 * hnr_score +
        0.1 * pitch_score
    )

    label = (
        "Likely Nervous" if nervousness_score >= 0.6 else
        "Somewhat Anxious" if nervousness_score >= 0.3 else
        "Calm"
    )

    return {
        "nervousness_score": round(nervousness_score, 2),
        "label": label
    }
