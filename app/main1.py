import os
import time
from datetime import datetime
from app.interview.tts import speak_text
from app.interview.llm_groq import generate_question
from app.interview.whisper_groq import transcribe_audio  # whisper_groq based
from app.interview.emotion import score_nervousness_relative
from app.interview.feature_extract import extract_voice_features, compute_relative_features
from app.interview.capture import record_audio
from app.interview.audio_utils import play_audio  # Utility to play audio cross-platform

# Setup session folder
def create_session_folder():
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    path = os.path.join("data", "interviews", session_id)
    os.makedirs(path, exist_ok=True)
    return path

# Main interview loop
def interview_loop(user_name: str, user_role: str, num_questions=3):
    session_path = create_session_folder()
    print(f"[👤] Hello {user_name}, we will ask you {num_questions} questions for the '{user_role}' role.\n")

    # --- Greeting ---
    greeting = f"Hello {user_name}, welcome to your mock interview for the role of {user_role}. Let's get started!"
    print(f"[AI]: {greeting}")
    greeting_path = os.path.join(session_path, "ai_greeting.wav")
    speak_text(greeting, greeting_path)
    play_audio(greeting_path)

    # --- Interview rounds ---
    previous_answers = []
    for i in range(1, num_questions + 1):
        print(f"\n🌀 Starting Round {i}...")

        # Generate LLM question with difficulty context
        difficulty = ["very short", "short", "moderate", "hard", "very hard"]
        context = (
            f"You are an interviewer for the position of {user_role}. "
            f"Generate a {difficulty[min(i-1, 4)]} question. "
            "Try to make the next question based on the previous answers if available. "
            f"Previous answers: {previous_answers if previous_answers else 'None'}"
        )

        question = generate_question(context)
        print(f"[🤖 AI Question]: {question}")
        q_audio_path = os.path.join(session_path, f"ai_question_{i}.wav")
        speak_text(question, q_audio_path)
        play_audio(q_audio_path)

        # Record user response
        print("[🎙️] Please answer (30 seconds)...")
        answer_audio_path = os.path.join(session_path, f"user_answer_{i}.wav")
        record_audio(answer_audio_path, duration=30)

        # Transcribe
        try:
            print("[🧠] Transcribing...")
            transcript = transcribe_audio(answer_audio_path)
            print(f"[📄] You said: {transcript}")
            with open(os.path.join(session_path, f"transcript_{i}.txt"), "w") as f:
                f.write(transcript)
            previous_answers.append(transcript)
        except Exception as e:
            print(f"[❌] Transcription failed: {e}")
            previous_answers.append("")

        # Voice features & emotion
        try:
            print("[📊] Extracting features...")
            features = extract_voice_features(answer_audio_path)
            rel_features = compute_relative_features(features)
            emotion = score_nervousness_relative(rel_features)
            print(f"[🧘] Nervousness Score: {emotion}")
        except Exception as e:
            print(f"[⚠️] Feature or emotion analysis failed: {e}")

    print("\n✅ Interview complete.")

# --- Entry Point ---
if __name__ == "__main__":
    print("🚀 AI Interview System")
    user_name = input("Enter your name: ").strip()
    user_role = input("Enter the job role you're applying for: ").strip()
    interview_loop(user_name, user_role, num_questions=3)
