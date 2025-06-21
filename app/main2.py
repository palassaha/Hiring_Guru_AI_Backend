import os
import time
from datetime import datetime
from app.interview.tts import speak_text
from app.interview.whisper_groq import transcribe_audio
from app.interview.capture import record_audio
from app.interview.audio_utils import play_audio
from app.communication.questions import CommunicationQuestionGenerator  # Import our generator

# Setup session folder
def create_session_folder():
    session_id = datetime.now().strftime("communication_%Y%m%d_%H%M%S")
    path = os.path.join("data", "communication", session_id)
    os.makedirs(path, exist_ok=True)
    return path

# Communication practice loop
def communication_practice_loop(user_name: str, practice_type: str, num_questions=5):
    session_path = create_session_folder()
    
    # Initialize the communication generator
    API_KEY = os.getenv("GROQ_API_KEY")  # Make sure to set this environment variable
    if not API_KEY:
        print("[❌] Please set GROQ_API_KEY environment variable")
        return
    
    generator = CommunicationQuestionGenerator()
    
    print(f"[👤] Hello {user_name}, let's practice {practice_type} communication!\n")
    
    # --- Greeting ---
    greeting = f"Hello {user_name}, welcome to your communication practice session. We'll work on {practice_type} today. Let's begin!"
    print(f"[AI]: {greeting}")
    greeting_path = os.path.join(session_path, "ai_greeting.wav")
    speak_text(greeting, greeting_path)
    play_audio(greeting_path)
    
    # Generate questions based on practice type
    questions = []
    
    if practice_type == "basic_sentences":
        print("[🧠] Generating basic sentences for you to practice...")
        sentences = generator.generate_basic_sentences(count=num_questions, difficulty="beginner")
        questions = [f"Please repeat this sentence clearly: {sentence}" for sentence in sentences]
        
    elif practice_type == "conversation":
        print("[🧠] Generating conversation questions...")
        questions = generator.generate_conversation_questions(count=num_questions, category="general")
        
    elif practice_type == "speaking_prompts":
        print("[🧠] Generating speaking prompts...")
        prompts = generator.generate_speaking_prompts(count=num_questions, time_limit="1 minute")
        questions = prompts
        
    elif practice_type == "comprehension":
        print("[🧠] Generating reading comprehension...")
        passage = generator.generate_comprehension_passage(topic="daily life", difficulty="intermediate")
        comp_questions = generator.generate_comprehension_questions(passage, count=num_questions)
        
        # First, read the passage
        passage_text = f"Here is a passage for you to listen to: {passage}"
        print(f"[📖] Passage: {passage}")
        passage_path = os.path.join(session_path, "passage.wav")
        speak_text(passage_text, passage_path)
        play_audio(passage_path)
        
        # Use comprehension questions
        questions = []
        for q in comp_questions.get("multiple_choice", []):
            options_text = " ".join([f"Option {chr(65+i)}: {opt}. " for i, opt in enumerate(q['options'])])
            questions.append(f"{q['question']} {options_text}")
        
        for q in comp_questions.get("short_answer", []):
            questions.append(q)
    
    if not questions:
        print("[❌] Failed to generate questions. Please try again.")
        return
    
    # --- Practice rounds ---
    user_responses = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n🌀 Round {i}/{len(questions)}...")
        print(f"[🤖 AI Question]: {question}")
        
        # Speak the question
        q_audio_path = os.path.join(session_path, f"question_{i}.wav")
        speak_text(question, q_audio_path)
        play_audio(q_audio_path)
        
        # Determine recording duration based on practice type
        if practice_type == "basic_sentences":
            duration = 10  # Short for sentence repetition
            print("[🎙️] Please repeat the sentence (10 seconds)...")
        elif practice_type == "conversation":
            duration = 45  # Medium for conversation
            print("[🎙️] Please answer the question (45 seconds)...")
        elif practice_type == "speaking_prompts":
            duration = 60  # Longer for speaking prompts
            print("[🎙️] Please speak about this topic (1 minute)...")
        elif practice_type == "comprehension":
            duration = 30  # Medium for comprehension
            print("[🎙️] Please answer based on the passage (30 seconds)...")
        else:
            duration = 30
            print("[🎙️] Please respond (30 seconds)...")
        
        # Record user response
        answer_audio_path = os.path.join(session_path, f"user_answer_{i}.wav")
        record_audio(answer_audio_path, duration=duration)
        
        # Transcribe user response
        try:
            print("[🧠] Transcribing your response...")
            transcript = transcribe_audio(answer_audio_path)
            print(f"[📄] You said: {transcript}")
            
            # Save transcript
            with open(os.path.join(session_path, f"transcript_{i}.txt"), "w") as f:
                f.write(f"Question: {question}\n")
                f.write(f"Answer: {transcript}\n")
            
            user_responses.append({
                'question': question,
                'answer': transcript
            })
            
            # Provide simple feedback
            if practice_type == "basic_sentences":
                # For sentence repetition, we could check similarity
                feedback = "Good job practicing the sentence!"
            else:
                # General positive feedback
                feedback = "Thank you for your response!"
            
            print(f"[✅] {feedback}")
            feedback_path = os.path.join(session_path, f"feedback_{i}.wav")
            speak_text(feedback, feedback_path)
            play_audio(feedback_path)
            
        except Exception as e:
            print(f"[❌] Transcription failed: {e}")
            user_responses.append({
                'question': question,
                'answer': "Transcription failed"
            })
        
        # Short pause between questions
        time.sleep(1)
    
    # --- Session Summary ---
    print(f"\n✅ Communication practice complete!")
    
    # Save session summary
    summary_path = os.path.join(session_path, "session_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Communication Practice Session\n")
        f.write(f"User: {user_name}\n")
        f.write(f"Practice Type: {practice_type}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions Completed: {len(user_responses)}\n\n")
        
        for i, response in enumerate(user_responses, 1):
            f.write(f"Round {i}:\n")
            f.write(f"Question: {response['question']}\n")
            f.write(f"Answer: {response['answer']}\n\n")
    
    # Final encouragement
    final_message = f"Great job, {user_name}! You completed {len(user_responses)} questions. Keep practicing to improve your communication skills!"
    print(f"[🎉] {final_message}")
    final_path = os.path.join(session_path, "final_message.wav")
    speak_text(final_message, final_path)
    play_audio(final_path)

def show_menu():
    print("\n" + "="*50)
    print("🎯 COMMUNICATION PRACTICE SYSTEM")
    print("="*50)
    print("1. Basic Sentences Practice")
    print("2. Conversation Questions")
    print("3. Speaking Prompts")
    print("4. Reading Comprehension")
    print("5. Exit")
    print("="*50)

# --- Entry Point ---
if __name__ == "__main__":
    print("🚀 Communication Practice System")
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("[⚠️] Warning: GROQ_API_KEY environment variable not set.")
        print("Please set it using: export GROQ_API_KEY='your_api_key_here'")
        print("Or add it to your .env file")
    
    user_name = input("Enter your name: ").strip()
    if not user_name:
        user_name = "Student"
    
    while True:
        show_menu()
        choice = input("\nSelect practice type (1-5): ").strip()
        
        if choice == "1":
            practice_type = "basic_sentences"
            num_questions = int(input("How many sentences to practice? (default 5): ") or 5)
            
        elif choice == "2":
            practice_type = "conversation"
            num_questions = int(input("How many conversation questions? (default 5): ") or 5)
            
        elif choice == "3":
            practice_type = "speaking_prompts"
            num_questions = int(input("How many speaking prompts? (default 3): ") or 3)
            
        elif choice == "4":
            practice_type = "comprehension"
            num_questions = int(input("How many comprehension questions? (default 4): ") or 4)
            
        elif choice == "5":
            print("Goodbye! Keep practicing your communication skills! 👋")
            break
            
        else:
            print("Invalid choice. Please try again.")
            continue
        
        # Start the practice session
        communication_practice_loop(user_name, practice_type, num_questions)
        
        # Ask if user wants to continue
        continue_choice = input("\nWould you like another practice session? (y/n): ").strip().lower()
        if continue_choice != 'y' and continue_choice != 'yes':
            print("Goodbye! Keep practicing your communication skills! 👋")
            break