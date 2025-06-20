import os
import openai
from app.config import GROQ_API_KEY

# Set up Groq API key and base URL
openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"

def generate_question(role: str, previous_question: str = None, previous_answer: str = None) -> str:
    """
    Generate a concise interview question for the given role.
    Questions should increase in difficulty based on candidate responses.
    """
    try:
        if previous_question and previous_answer:
            prompt = (
                f"You are an AI interviewer conducting a mock interview for the role of '{role}'.\n"
                f"Your questions should be short or very short (no more than 15 words).\n"
                f"The difficulty should gradually increase with each question.\n"
                f"Previous question: {previous_question}\n"
                f"Candidate's answer: {previous_answer}\n"
                f"Now generate the next, slightly more difficult question based on this."
            )
        else:
            prompt = (
                f"You are an AI interviewer conducting a mock interview for the role of '{role}'.\n"
                f"Ask the first question. It should be short (under 10-15 words), clear, and easy."
            )

        response = openai.ChatCompletion.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a concise and adaptive mock interview assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        return text
    except Exception as e:
        print(f"[❌] LLM generation failed: {e}")
        return "Sorry, I couldn't generate a question."
