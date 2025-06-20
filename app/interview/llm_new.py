import os
import openai
from app.config import GROQ_API_KEY

# Configure Groq API
openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"

def generate_question(role: str, previous_question: str = None, previous_answer: str = None, emotion_label: str = None) -> str:
    """
    Generate a concise interview question.
    AI will ask 3 questions max. Each question becomes slightly more difficult.
    If emotion_label is 'nervous', question tone should be softer.
    """

    # System prompt
    system_prompt = (
        "You are an AI interviewer conducting a mock interview. "
        "Ask short (under 15 words), clear questions that increase in difficulty over time. "
        "If the user seems nervous, be a bit encouraging."
    )

    # Prompt based on state
    if previous_question and previous_answer and emotion_label:
        user_prompt = (
            f"Role: {role}\n"
            f"Previous Question: {previous_question}\n"
            f"User's Answer: {previous_answer}\n"
            f"Emotion: {emotion_label}\n\n"
            f"Now generate the next, slightly harder question for the same interview."
        )
    else:
        user_prompt = (
            f"You are starting a mock interview for the role of '{role}'. "
            f"Ask the first question. It should be short, clear, and easy."
        )

    try:
        response = openai.ChatCompletion.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[❌] LLM generation failed: {e}")
        return "Sorry, I couldn't generate a question."
