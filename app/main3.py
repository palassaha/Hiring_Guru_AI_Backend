import os
import json
import uuid
from datetime import datetime
from typing import Any, List, Literal, Optional
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from typing import Dict
import uvicorn
from fastapi import Form
from app.aptitude.result import generate_answer_groq, generate_detailed_feedback, is_answer_correct
from app.classes import AssessmentRequest, AssessmentResponse, AudioRequest, BasicSentencesRequest, BasicSentencesResponse, ComprehensionRequest, ComprehensionResponse, EvaluationRequest, EvaluationRequestTechnical, EvaluationResponse, GenerateAptitudeQuestionsRequest, GenerateTechnicalQuestionsRequest, MultipleChoiceQuestion, PronunciationCheckResponse, ScreeningRequest, ScreeningResponse, TranscriptionResponse
from app.dsa_coding.scraper_3 import scrape_random_questions
from app.technical.results import generate_answer_groq as generate_answer_groq_technical , generate_detailed_feedback as generate_detailed_feedback_technical, is_answer_correct as is_answer_correct_technical
from app.aptitude.scraper import AptitudeQuestionScraper
from app.communication.check import PronunciationScorer
from app.communication.comms import CommunicationQuestionGenerator
from app.interview.tts import speak_text
from app.interview.whisper_groq import transcribe_audio
from app.aptitude.llm import process_questions
from app.technical.llm import process_questions as process_technical_questions
from app.screening.screening import JobScreeningSystem
from app.technical.scraper import TechnicalQuestionScraper

# Load environment variables
load_dotenv()

# Configure OpenAI to use Groq's endpoint
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

# Initialize FastAPI app
app = FastAPI(
    title="Communication Practice API",
    description="API for generating communication practice materials",
    version="1.0.0"
)

# Pydantic models for request/response
class BasicSentencesRequest(BaseModel):
    count: int = 10
    difficulty: str = "beginner"

class BasicSentencesResponse(BaseModel):
    sentences: List[str]
    session_id: str

class AudioRequest(BaseModel):
    sentence: str
    session_id: Optional[str] = None

class ComprehensionRequest(BaseModel):
    topic: str = "daily life"
    difficulty: str = "intermediate"
    question_count: int = 5

class MultipleChoiceQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

class ComprehensionResponse(BaseModel):
    passage: str
    multiple_choice: List[MultipleChoiceQuestion]
    short_answer: List[str]
    session_id: str

class TranscriptionResponse(BaseModel):
    transcription: str

class PronunciationCheckRequest(BaseModel):
   original_sentence: str
   audio_file: Optional[str] = None  
   transcribed_text: Optional[str] = None  

class PronunciationCheckResponse(BaseModel):
   similarity_percentage: float
   original_text: str
   spoken_text: str
   feedback: str

class ScreeningRequest(BaseModel):
    company_with_role: str

class AssessmentRequest(BaseModel):
    company_with_role: str
    questions: List[Dict[str, Any]]
    responses: Dict[int, str]

class ScreeningResponse(BaseModel):
    company: str
    role: str
    role_title: str
    questions: List[Dict[str, Any]]
    scoring_criteria: Dict[str, int]
    generated_at: str
    total_questions: int

class AssessmentResponse(BaseModel):
    overall_score: int
    category_scores: Dict[str, int]
    strengths: List[str]
    areas_for_improvement: List[str]
    detailed_feedback: Dict[str, str]
    recommendation: str
    recommendation_reason: str
    next_steps: List[str]
    red_flags: List[str]
    standout_responses: List[str]
    assessment_date: str
    company: str
    role: str
    role_title: str
    total_responses: int
    response_completion_rate: float

class GenerateAptitudeQuestionsRequestModel(BaseModel):
    roundType: str  # "APTITUDE", etc.
    difficulty: str  # "easy", "medium", "hard"
    questionCount: int
    category: Optional[str] = None
    duration: int
    type: str  # "MCQ", "SUBJECTIVE", etc.


class GenerateAptitudeQuestionsRequest(BaseModel):
    questions_with_answers: List[dict]
# Communication Question Generator Class
class GenerateTechnicalQuestionsRequest(BaseModel):
    questions_with_answers: List[dict]

# Initialize generator
generator = CommunicationQuestionGenerator()
pronunciation_scorer = PronunciationScorer()
screening= JobScreeningSystem()


# Create directories for storing files
os.makedirs("data/audio", exist_ok=True)
os.makedirs("data/sessions", exist_ok=True)

# Helper function to create session ID
def create_session_id():
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Communication Practice API is running!"}

@app.post("/api/basic-sentences", response_model=BasicSentencesResponse)
async def get_basic_sentences(request: BasicSentencesRequest):
    """
    API 1: Generate basic English sentences for practice
    """
    try:
        # Validate difficulty level
        valid_difficulties = ["beginner", "intermediate", "advanced"]
        if request.difficulty not in valid_difficulties:
            raise HTTPException(status_code=400, detail=f"Difficulty must be one of: {valid_difficulties}")
        
        # Validate count
        if request.count < 1 or request.count > 50:
            raise HTTPException(status_code=400, detail="Count must be between 1 and 50")
        
        # Generate sentences
        sentences = generator.generate_basic_sentences(request.count, request.difficulty)
        
        if not sentences:
            raise HTTPException(status_code=500, detail="Failed to generate sentences")
        
        # Create session ID
        session_id = create_session_id()
        
        # Save session data
        session_data = {
            "session_id": session_id,
            "type": "basic_sentences",
            "difficulty": request.difficulty,
            "count": request.count,
            "sentences": sentences,
            "created_at": datetime.now().isoformat()
        }
        
        session_file = f"data/sessions/{session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
        
        return BasicSentencesResponse(
            sentences=sentences,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sentences: {str(e)}")

@app.post("/api/sentence-audio")
async def get_sentence_audio(request: AudioRequest):
    """
    API 2: Generate WAV file for a given sentence
    """
    try:
        if not request.sentence.strip():
            raise HTTPException(status_code=400, detail="Sentence cannot be empty")
        
        # Create unique filename
        audio_id = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        audio_path = f"data/audio/{audio_id}.wav"
        
        # Generate TTS audio
        speak_text(request.sentence, audio_path)
        
        # Check if file was created
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate audio file")
        
        # Return the audio file
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"{audio_id}.wav",
            headers={"Content-Disposition": f"attachment; filename={audio_id}.wav"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.post("/api/comprehension", response_model=ComprehensionResponse)
async def get_comprehension_material(request: ComprehensionRequest):
    """
    API 3: Generate reading comprehension passage and questions
    """
    try:
        # Validate difficulty level
        valid_difficulties = ["beginner", "intermediate", "advanced"]
        if request.difficulty not in valid_difficulties:
            raise HTTPException(status_code=400, detail=f"Difficulty must be one of: {valid_difficulties}")
        
        # Validate question count
        if request.question_count < 1 or request.question_count > 20:
            raise HTTPException(status_code=400, detail="Question count must be between 1 and 20")
        
        # Generate passage
        passage = generator.generate_comprehension_passage(request.topic, request.difficulty)
        
        if not passage:
            raise HTTPException(status_code=500, detail="Failed to generate passage")
        
        # Generate questions
        questions_data = generator.generate_comprehension_questions(passage, request.question_count)
        
        # Create session ID
        session_id = create_session_id()
        
        # Format multiple choice questions
        multiple_choice = []
        for q in questions_data.get("multiple_choice", []):
            multiple_choice.append(MultipleChoiceQuestion(
                question=q["question"],
                options=q["options"],
                correct_answer=q["correct_answer"]
            ))
        
        short_answer = questions_data.get("short_answer", [])
        
        # Save session data
        session_data = {
            "session_id": session_id,
            "type": "comprehension",
            "topic": request.topic,
            "difficulty": request.difficulty,
            "question_count": request.question_count,
            "passage": passage,
            "multiple_choice": [q.dict() for q in multiple_choice],
            "short_answer": short_answer,
            "created_at": datetime.now().isoformat()
        }
        
        session_file = f"data/sessions/{session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
        
        return ComprehensionResponse(
            passage=passage,
            multiple_choice=multiple_choice,
            short_answer=short_answer,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating comprehension material: {str(e)}")

# Fixed audio transcription endpoint
@app.post("/api/transcribe-audio", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """
    API 4: Transcribe audio file to text
    """
    print("Audio transcription request received")
    temp_audio_path = None
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read the uploaded file
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="No audio file content provided")
        
        # Save the uploaded audio file temporarily
        temp_audio_path = f"data/audio/temp_{uuid.uuid4().hex}.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(file_content)
        
        # Transcribe the audio
        transcription = transcribe_audio(temp_audio_path)
        print(f"[📄] Transcription: {transcription}")
        
        if not transcription:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")
        
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return TranscriptionResponse(transcription=transcription)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise
    except Exception as e:
        # Clean up temporary file on error
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

@app.post("/api/check-pronunciation", response_model=PronunciationCheckResponse)
async def check_pronunciation(
   original_sentence: str = Form(...),
   audio_file: UploadFile = File(...)
):
   """
   API: Check pronunciation accuracy by comparing original sentence with spoken audio
   """
   temp_audio_path = None
   
   try:
       # Validate inputs
       if not original_sentence.strip():
           raise HTTPException(status_code=400, detail="Original sentence cannot be empty")
       
       if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
           raise HTTPException(status_code=400, detail="File must be an audio file")
       
       # Save uploaded audio temporarily
       file_content = await audio_file.read()
       temp_audio_path = f"data/audio/temp_pronunciation_{uuid.uuid4().hex}.wav"
       
       with open(temp_audio_path, "wb") as f:
           f.write(file_content)
       
       # Transcribe the audio
       transcribed_text = transcribe_audio(temp_audio_path)
       
       if not transcribed_text:
           raise HTTPException(status_code=500, detail="Failed to transcribe audio")
       
       # Calculate similarity
       result = pronunciation_scorer.calculate_similarity(original_sentence, transcribed_text)
       
       # Generate feedback based on similarity score
       similarity_score = result["similarity_percentage"]
       if similarity_score >= 90:
           feedback = "Excellent pronunciation! Very close to the original."
       elif similarity_score >= 80:
           feedback = "Good pronunciation! Minor differences detected."
       elif similarity_score >= 70:
           feedback = "Fair pronunciation. Some words may need practice."
       elif similarity_score >= 50:
           feedback = "Needs improvement. Several words were not clear."
       else:
           feedback = "Requires significant practice. Most words were unclear."
       
       # Clean up temporary file
       if os.path.exists(temp_audio_path):
           os.remove(temp_audio_path)
       
       return PronunciationCheckResponse(
           similarity_percentage=similarity_score,
           original_text=original_sentence,
           spoken_text=transcribed_text,
           feedback=feedback
       )
       
   except HTTPException:
       if temp_audio_path and os.path.exists(temp_audio_path):
           os.remove(temp_audio_path)
       raise
   except Exception as e:
       if temp_audio_path and os.path.exists(temp_audio_path):
           os.remove(temp_audio_path)
       raise HTTPException(status_code=500, detail=f"Error checking pronunciation: {str(e)}")


@app.post("/api/generate-aptitude-questions")
async def generate_aptitude_questions(req: GenerateAptitudeQuestionsRequestModel):
    try:
        # 1. Scrape questions
        scraper = AptitudeQuestionScraper(headless=True)
        scraped_questions = scraper.run_scraping()

        if not scraped_questions:
            raise HTTPException(status_code=500, detail="No questions were scraped from the websites")

        # 2. Process scraped questions via AI
        try:
            process_questions("app/aptitude/question_bank.json", "app/aptitude/questions_with_answers.json")

            with open("app/aptitude/questions_with_answers.json", "r", encoding="utf-8") as f:
                questions_with_answers = json.load(f)

            if not questions_with_answers:
                questions_with_answers = scraped_questions

        except Exception as ai_error:
            print(f"AI processing error: {ai_error}")
            questions_with_answers = scraped_questions

        # 3. Trim/filter questions if needed
        trimmed_questions = questions_with_answers[:req.questionCount]

        return {
            "questions_with_answers": trimmed_questions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating aptitude questions: {str(e)}")

@app.get("/api/generate-technical-questions", response_model=GenerateTechnicalQuestionsRequest)
async def generate_technical_questions():
    """
    API 5: Generate technical questions using web scraping
    """
    try:
        # Create scraper instance and run scraping
        scraper = TechnicalQuestionScraper(headless=True)
        scraped_questions = scraper.run_scraping()
        
        if not scraped_questions:
            raise HTTPException(status_code=500, detail="No questions were scraped from the websites")
        
        # Process questions with AI to generate answers
        try:
            process_technical_questions("app/technical/question_bank.json", "app/technical/questions_with_answers.json")
            
            # Load the processed questions with answers
            with open("app/technical/questions_with_answers.json", "r", encoding="utf-8") as f:
                questions_with_answers = json.load(f)
                
            if not questions_with_answers:
                # If AI processing failed, return the scraped questions without answers
                print("Warning: AI processing failed, returning scraped questions without answers")
                questions_with_answers = scraped_questions
                
        except Exception as ai_error:
            print(f"AI processing error: {ai_error}")
            # Return scraped questions without AI-generated answers
            questions_with_answers = scraped_questions
        
        return GenerateTechnicalQuestionsRequest(questions_with_answers=questions_with_answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating aptitude questions: {str(e)}")

@app.post("/generate-questions", response_model=ScreeningResponse)
async def generate_screening_questions(request: ScreeningRequest):
    """
    Generate screening questions for any company and role
    
    Example: {"company_with_role": "Amazon SDE 1"}
    """
    try:
        questions_data = await screening.generate_screening_questions(request.company_with_role)
        return ScreeningResponse(**questions_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.post("/assess-responses", response_model=AssessmentResponse)
async def assess_candidate_responses(request: AssessmentRequest):
    """
    Assess candidate responses and provide detailed feedback
    
    Example:
    {
        "company_with_role": "Amazon SDE 1",
        "questions": [...], // questions from generate-questions endpoint
        "responses": {
            1: "I have 2 years of experience...",
            2: "I'm interested in Amazon because..."
        }
    }
    """
    try:
        assessment = await screening.assess_candidate_responses(
            request.company_with_role, 
            request.questions, 
            request.responses
        )
        return AssessmentResponse(**assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assessing responses: {str(e)}")

@app.post("/evaluate-aptitude-answers", response_model=EvaluationResponse)
async def evaluate_answers(request: EvaluationRequest):
    """
    Evaluate aptitude test answers using LLM-generated correct answers
    """
    try:
        results = []
        correct_answers = 0
        
        for question_data in request.questions:
            # Generate correct answer using LLM
            correct_answer = generate_answer_groq(question_data.question, question_data.options)
            
            if not correct_answer:
                raise HTTPException(status_code=500, detail=f"Failed to generate answer for question: {question_data.question[:50]}...")
            
            # Check if user answer is correct
            is_correct = is_answer_correct(question_data.answer, correct_answer)
            
            if is_correct:
                correct_answers += 1
            
            result = {
                "question": question_data.question,
                "user_answer": question_data.answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "options": question_data.options
            }
            results.append(result)
        
        # Calculate overall score
        overall_score = (correct_answers / len(request.questions)) * 100 if request.questions else 0
        
        # Generate feedback
        feedback = generate_detailed_feedback(results, overall_score)
        
        return EvaluationResponse(
            overallScore=f"{round(overall_score, 1)}%",
            feedback=feedback,
            detailedResults=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing evaluation: {str(e)}")

@app.post("/api/evaluate-technical-answers", response_model=EvaluationResponse)
async def evaluate_technical_answers(request: EvaluationRequestTechnical):
    """
    Evaluate answers using LLM-generated correct answers for different question types
    """
    try:
        results = []
        correct_answers = 0
        
        for question_data in request.questions:
            # Generate correct answer using LLM based on question type
            correct_answer = generate_answer_groq_technical(
                question_data.question, 
                question_data.options, 
                request.question_type
            )
            
            if not correct_answer:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to generate answer for question: {question_data.question[:50]}..."
                )
            
            # Check if user answer is correct
            is_correct = is_answer_correct_technical(question_data.answer, correct_answer)
            
            if is_correct:
                correct_answers += 1
            
            result = {
                "question": question_data.question,
                "user_answer": question_data.answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "options": question_data.options
            }
            results.append(result)
        
        # Calculate overall score
        overall_score = (correct_answers / len(request.questions)) * 100 if request.questions else 0
        
        # Generate feedback based on question type
        feedback = generate_detailed_feedback_technical(results, overall_score, request.question_type)
        
        return EvaluationResponse(
            overallScore=f"{round(overall_score, 1)}%",
            feedback=feedback,
            detailedResults=results,
            questionType=request.question_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing evaluation: {str(e)}")

@app.get("/questions")
async def get_questions():
    """
    Get random LeetCode questions and return as JSON
    Scrapes 2 Easy + 2 Medium + 1 Hard questions
    
    Returns:
        dict: Questions data as JSON
    """
    try:
        # Check if saved file exists first
        file_paths = [
            "./app/dsa_coding/random_leetcode_questions.json",
            "random_leetcode_questions.json"
        ]
        
        # Try to load existing file
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # If no file exists, scrape new questions
        results = scrape_random_questions()
        
        if 'error' in results:
            raise HTTPException(status_code=500, detail=f"Failed to scrape questions: {results['error']}")
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_key_configured": bool(openai.api_key)
    }

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    try:
        session_file = f"data/sessions/{session_id}.json"
        
        if not os.path.exists(session_file):
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(session_file, "r") as f:
            session_data = json.load(f)
        
        return session_data
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

if __name__ == "__main__":
    # Check for API key
    if not openai.api_key:
        print("[❌] GROQ_API_KEY environment variable not set!")
        print("Please set it using: export GROQ_API_KEY='your_api_key_here'")
        print("Or add it to your .env file")
        exit(1)
    
    print("🚀 Starting Communication Practice API Server...")
    print("📝 API Documentation will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main3:app",  # Change this to your actual filename if different
        host="0.0.0.0",
        port=8000,
        reload=True
    )