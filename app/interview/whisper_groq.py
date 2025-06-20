import requests
import os
from app.config import GROQ_API_KEY

GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

def transcribe_audio(file_path: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

    with open(file_path, "rb") as audio_file:
        response = requests.post(
            GROQ_WHISPER_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": audio_file},
            data={"model": "whisper-large-v3"}
        )

    if response.status_code == 200:
        return response.json()["text"]
    else:
        raise Exception(f"Whisper API error {response.status_code}: {response.text}")
