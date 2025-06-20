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
from app.interview.llm_engine import generate_next_question
from app.interview.tts import speak_text
from app.interview.audio_utils import play_audio



def run_interview_session(duration=10):
    print("[🎥] Starting interview session (audio + video)...")
    session_folder = record_audio_video(duration=duration)

    audio_path = os.path.join(session_folder, "user_audio.wav")
    print(f"[🔊] Audio saved to: {audio_path}")

    # --- Step 1: Transcription ---
    try:
        print("[🧠] Transcribing with Groq Whisper...")
        transcript = transcribe_audio(audio_path)
        print(f"[📝] Transcript: {transcript}")

        with open(os.path.join(session_folder, "transcript.txt"), "w") as f:
            f.write(transcript)
    except Exception as e:
        print(f"[❌] Transcription failed: {e}")
        return

    # --- Step 2: Voice Feature Extraction ---
    try:
        print("[📊] Extracting voice features...")
        features = extract_voice_features(audio_path)
        print("[📈] Features:", features)

        with open(os.path.join(session_folder, "features.txt"), "w") as f:
            for k, v in features.items():
                f.write(f"{k}: {v}\n")
    except Exception as e:
        print(f"[❌] Feature extraction failed: {e}")
        return

    # --- Step 3: Relative Feature Changes ---
    try:
        print("[📉] Loading baseline...")
        baseline = load_baseline()
        relative = compute_relative_features(features, baseline)
        print("[📉] Relative Feature Changes:", relative)

        with open(os.path.join(session_folder, "relative_features.txt"), "w") as f:
            for k, v in relative.items():
                f.write(f"{k}_delta: {v}\n")
    except Exception as e:
        print(f"[⚠️] Could not compute relative features: {e}")
        relative = {}

    # --- Step 4: Emotion/Nervousness Detection ---
    try:
        print("[🧠] Analyzing emotional cues...")
        emotion = score_nervousness_relative(relative)
        print("[🧘] Nervousness Score:", emotion)

        with open(os.path.join(session_folder, "emotion.json"), "w") as f:
            json.dump(emotion, f, indent=2)
    except Exception as e:
        print(f"[❌] Emotion detection failed: {e}")
        emotion = {"label": "Unknown"}

    # --- Step 5: LLM-generated Response via Groq ---
    try:
        print("[🤖] Generating next question with Groq LLM...")
        llm_response = generate_next_question(transcript, emotion.get("label", "Unknown"))
        print("[💬] AI says:", llm_response)

        with open(os.path.join(session_folder, "llm_response.txt"), "w") as f:
            f.write(llm_response)
    except Exception as e:
        print(f"[❌] LLM generation failed: {e}")
        llm_response = "Sorry, I couldn't generate a question."

    # --- Step 6: TTS (Text to Speech) ---
    try:
        print("[🔊] Converting AI response to speech...")
        tts_output = os.path.join(session_folder, "ai_response.wav")
        speak_text(llm_response, tts_output)
    except Exception as e:
        print(f"[❌] TTS synthesis failed: {e}")
    
    
    # --- Step 7: Auto-play TTS output ---
    try:
        play_audio(tts_output)
    except Exception as e:
        print(f"[❌] Playback failed: {e}")

    print(f"[✅] Interview session complete: {session_folder}")


if __name__ == "__main__":
    run_interview_session(duration=10)
