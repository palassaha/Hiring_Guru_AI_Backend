import os
import json

from app.interview.capture import record_audio_video
from app.interview.whisper_groq import transcribe_audio
from app.interview.feature_extract import (
    extract_voice_features,
    compute_relative_features,
    load_baseline
)
from app.interview.emotion_detector import score_nervousness_relative

def run_interview_session(duration=10):
    print("[🎥] Starting interview session (audio + video)...")
    session_folder = record_audio_video(duration=duration)

    audio_path = os.path.join(session_folder, "user_audio.wav")
    print(f"[🔊] Audio saved to: {audio_path}")

    # ---- Step 1: Transcribe ----
    try:
        print("[🧠] Transcribing with Groq Whisper...")
        transcript = transcribe_audio(audio_path)
        print(f"[📝] Transcript: {transcript}")

        transcript_path = os.path.join(session_folder, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write(transcript)
        print(f"[💾] Transcript saved at: {transcript_path}")
    except Exception as e:
        print(f"[❌] Transcription failed: {e}")
        return

    # ---- Step 2: Feature Extraction ----
    try:
        print("[📊] Extracting voice features...")
        features = extract_voice_features(audio_path)
        print("[📈] Features:", features)

        features_path = os.path.join(session_folder, "features.txt")
        with open(features_path, "w") as f:
            for k, v in features.items():
                f.write(f"{k}: {v}\n")
        print(f"[💾] Features saved at: {features_path}")
    except Exception as e:
        print(f"[❌] Feature extraction failed: {e}")
        return

    # ---- Step 3: Load Baseline + Compute Deltas ----
    try:
        print("[📉] Loading baseline features...")
        baseline = load_baseline()
        relative = compute_relative_features(features, baseline)
        print("[📉] Relative Feature Changes:", relative)

        relative_path = os.path.join(session_folder, "relative_features.txt")
        with open(relative_path, "w") as f:
            for k, v in relative.items():
                f.write(f"{k}_delta: {v}\n")
    except Exception as e:
        print(f"[⚠️] Could not compute relative features: {e}")
        relative = {}

    # ---- Step 4: Emotion/Nervousness Scoring ----
    try:
        print("[🧠] Analyzing emotional cues...")
        emotion = score_nervousness_relative(relative)
        print("[🧘] Nervousness Score:", emotion)

        emotion_path = os.path.join(session_folder, "emotion.json")
        with open(emotion_path, "w") as f:
            json.dump(emotion, f, indent=2)
        print(f"[💾] Emotion score saved at: {emotion_path}")
    except Exception as e:
        print(f"[❌] Emotion detection failed: {e}")

    print(f"[✅] Interview session complete: {session_folder}")
    return session_folder

if __name__ == "__main__":
    run_interview_session(duration=10)
