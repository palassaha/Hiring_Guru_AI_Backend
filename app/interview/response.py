from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import time
import uuid
from datetime import datetime
import json

# Import your existing modules
from app.interview.tts import speak_text
from app.interview.llm_groq import generate_question
from app.interview.whisper_groq import transcribe_audio
from app.interview.emotion import score_nervousness_relative
from app.interview.feature_extract import extract_voice_features, compute_relative_features



# Pydantic models for request/response
class GreetingRequest(BaseModel):
    user_name: str
    user_role: str

class GreetingResponse(BaseModel):
    session_id: str
    greeting_text: str
    audio_file_path: str

class AIResponseRequest(BaseModel):
    session_id: str
    question_number: Optional[int] = 1
    user_transcript: Optional[str] = None

class AIResponseResponse(BaseModel):
    question_text: str
    audio_file_path: str
    question_number: int
    session_id: str

class AnalysisResponse(BaseModel):
    transcript: str
    nervousness_score: Optional[float] = None
    features: Optional[dict] = None

# In-memory session storage (in production, use Redis or database)


