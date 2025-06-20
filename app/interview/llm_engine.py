import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def generate_next_question(transcript: str, emotion_label: str) -> str:
    system_prompt = (
        "You are an AI interviewer assisting in a hiring process. "
        "Based on the candidate's emotional state and previous answer, "
        "you ask appropriate follow-up questions. If they seem nervous, offer reassurance."
    )

    user_prompt = f"""
Transcript: "{transcript}"
Emotion: "{emotion_label}"

Generate the next interview question. Keep it concise and helpful.
"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[❌] LLM generation failed: {e}")
        return "Sorry, I couldn't generate a question."
