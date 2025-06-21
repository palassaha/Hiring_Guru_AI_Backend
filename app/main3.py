import base64
import os
import json
import uuid
from datetime import datetime
from typing import Any, List, Literal, Optional
from bson import ObjectId
from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile, File, logger
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from typing import Dict
import uvicorn
from fastapi import Form
from app.aptitude.result import generate_answer_groq, generate_detailed_feedback, is_answer_correct
from app.classes import AIResponseRequest, AnalysisResponse, AssessmentRequest, AssessmentResponse, AudioRequest, BasicSentencesRequest, BasicSentencesResponse, ComprehensionRequest, ComprehensionResponse, EvaluationRequest, EvaluationRequestTechnical, EvaluationResponse, GenerateAptitudeQuestionsRequest, GenerateAptitudeQuestionsRequestModel, GenerateTechnicalQuestionsRequest, GreetingRequest, MultipleChoiceQuestion, PronunciationCheckResponse, ScreeningRequest, ScreeningResponse, SessionScoreResponse, TechnicalGenerationInput, TranscriptionResponse, UserId, UserProfileRequest
from app.dsa_coding.boiler_plate import generate_boilerplates_async
from app.dsa_coding.scraper_3 import scrape_random_questions
from app.interview.emotion_detector import score_nervousness_relative
from app.interview.feature_extract import compute_relative_features, extract_voice_features
from app.interview.llm_groq import generate_question
from app.interview.scores import SessionAnalyzer
from app.mongo import InterviewDataManager, MongoDBHandler
from app.onboarding.onboard import AssessmentGenerator
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
import httpx

load_dotenv()

# Configure OpenAI to use Groq's endpoint
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"
dburl = os.getenv("DATABASE_URL")  # Default to local MongoDB if not set
db_handler = MongoDBHandler(
    connection_string=dburl,  # Update with your connection string
    database_name="h4b-2025"
)

# Connect to MongoDB
if not db_handler.connect():
    raise Exception("Failed to connect to MongoDB")
interview_manager = InterviewDataManager(db_handler)
# Initialize FastAPI app
app = FastAPI(
    title="Communication Practice API",
    description="API for generating communication practice materials",
    version="1.0.0"
)

# Pydantic models for request/response

# Initialize generator
generator = CommunicationQuestionGenerator()
pronunciation_scorer = PronunciationScorer()
screening= JobScreeningSystem()
session_analyzer = SessionAnalyzer(groq_api_key=os.getenv("GROQ_API_KEY"))

# Create directories for storing files
os.makedirs("data/audio", exist_ok=True)
os.makedirs("data/sessions", exist_ok=True)

# Helper function to create session ID
def create_session_id():
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

sessions = {}

def create_session_folder(session_id: str):
    """Create session folder for storing audio files"""
    path = os.path.join("data", "interviews", session_id)
    os.makedirs(path, exist_ok=True)
    return path

def get_session(session_id: str):
    """Get session data"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

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

@app.post("/api/generate-technical-questions", response_model=GenerateTechnicalQuestionsRequest)
async def generate_technical_questions(input: TechnicalGenerationInput):
    try:
        # Scrape questions (you can also filter based on input.category etc.)
        scraper = TechnicalQuestionScraper(headless=True)
        scraped_questions = scraper.run_scraping()

        if not scraped_questions:
            raise HTTPException(status_code=500, detail="No questions were scraped from the websites")

        try:
            process_technical_questions(
                "app/technical/question_bank.json",
                "app/technical/questions_with_answers.json"
            )

            with open("app/technical/questions_with_answers.json", "r", encoding="utf-8") as f:
                questions_with_answers = json.load(f)

            if not questions_with_answers:
                print("AI processing failed, using scraped questions")
                questions_with_answers = scraped_questions

        except Exception as ai_error:
            print(f"AI processing error: {ai_error}")
            questions_with_answers = scraped_questions

        # Optionally, trim based on `input.questionCount`
        return GenerateTechnicalQuestionsRequest(
            questions_with_answers=questions_with_answers[:input.questionCount]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating technical questions: {str(e)}")


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

@app.post("/api/evaluate-aptitude-answers", response_model=EvaluationResponse)
async def evaluate_answers(request: EvaluationRequest):
    """
    Evaluate aptitude test answers using LLM-generated correct answers
    """
    try:
        results = []
        correct_answers = 0
        
        for question_data in request.questions:
            # Generate correct answer using LLM
            correct_answer = generate_answer_groq(question_data.question)
            print(f"Generated answer for question: {question_data.question[:50]}... -> {correct_answer}")
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
    Get random LeetCode questions with generated boilerplates and return as JSON
    Processes questions from all_random_questions.json and generates code templates
    
    Returns:
        dict: Structured questions data with boilerplates as JSON
    """
    try:
        # Check if processed file exists first
        processed_file_paths = [
            "./app/dsa_coding/processed_questions_with_boilerplates.json",
            "processed_questions_with_boilerplates.json"
        ]
        
        # Try to load existing processed file
        for file_path in processed_file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # If no processed file exists, generate boilerplates
        input_file_paths = [
            "./app/dsa_coding/all_random_questions.json",
            "all_random_questions.json"
        ]
        
        input_file = None
        for file_path in input_file_paths:
            if os.path.exists(file_path):
                input_file = file_path
                break
        
        if not input_file:
            raise HTTPException(status_code=404, detail="Source questions file not found")
        
        # Generate structured output with boilerplates
        structured_output = await generate_boilerplates_async(input_file)
        
        # Save the processed output for future requests
        save_path = "./app/dsa_coding/processed_questions_with_boilerplates.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(structured_output, f, indent=2)
        
        return structured_output
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/greeting")
async def create_greeting(request: GreetingRequest):
    """
    Create a greeting for the user and initialize interview session
    Returns JSON with greeting text and audio data
    """
    try:
        # Generate unique session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Generate round_id if not provided
        if not request.round_id:
            round_id = str(ObjectId())
        else:
            round_id = request.round_id
        
        # Create session folder
        session_path = create_session_folder(session_id)
        
        # Generate greeting text
        greeting_text = (
            f"Hello {request.user_name}, welcome to your mock interview "
            f"for the role of {request.user_role}. Let's get started!"
        )
        
        # Generate greeting audio
        greeting_audio_path = os.path.join(session_path, "ai_greeting.wav")
        speak_text(greeting_text, greeting_audio_path)
        
        # Check if file was created
        if not os.path.exists(greeting_audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate greeting audio")
        
        # MONGODB INTEGRATION: Store greeting in database
        auth_token = f"auth_{str(uuid.uuid4())[:16]}"  # Generate auth token
        db_success = interview_manager.handle_greeting_creation(
            session_id=session_id,
            user_name=request.user_name,
            user_role=request.user_role,
            greeting_text=greeting_text,
            round_id=round_id,
            auth_token=auth_token,
            greeting_audio_path=greeting_audio_path
        )
        
        if not db_success:
            logger.warning(f"Failed to store greeting in database for session {session_id}")
        
        # Initialize session data (keep existing in-memory storage for compatibility)
        sessions[session_id] = {
            "user_name": request.user_name,
            "user_role": request.user_role,
            "session_path": session_path,
            "previous_answers": [],
            "current_question": 0,
            "created_at": datetime.now().isoformat(),
            "round_id": round_id,
            "auth_token": auth_token,
            "db_stored": db_success
        }
        
        # Read audio file and encode as base64
        with open(greeting_audio_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Return JSON response
        return {
            "session_id": session_id,
            "round_id": round_id,
            "greeting_text": greeting_text,
            "audio_data": audio_data,
            "audio_format": "wav",
            "message": "Greeting created successfully",
            "db_stored": db_success
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create greeting: {str(e)}")

@app.post("/ai-response")
async def generate_ai_response(request: AIResponseRequest):
    """
    Generate AI question based on session context and previous answers
    Returns JSON with question text and audio data
    """
    try:
        # Get session data
        session = get_session(request.session_id)
        session_path = session["session_path"]
        
        # Update question number
        if request.question_number:
            session["current_question"] = request.question_number
        else:
            session["current_question"] += 1
        
        current_question_num = session["current_question"]
        
        # Add user transcript to previous answers if provided
        if request.user_transcript and request.user_transcript.strip():
            session["previous_answers"].append({
                "question_number": current_question_num - 1,
                "answer": request.user_transcript.strip()
            })
        
        # Generate question context with better structure
        difficulty_levels = ["basic", "intermediate", "moderate", "advanced", "expert"]
        difficulty = difficulty_levels[min(current_question_num - 1, 4)]
        
        # Build context for question generation
        previous_qa_context = ""
        if session["previous_answers"]:
            previous_qa_context = "Previous Q&A in this interview:\n"
            for i, qa in enumerate(session["previous_answers"][-3:], 1):
                previous_qa_context += f"Q{qa.get('question_number', i)}: [Previous question]\n"
                previous_qa_context += f"A{qa.get('question_number', i)}: {qa['answer']}\n\n"
        
        context = (
            f"You are conducting a mock interview for a {session['user_role']} position. "
            f"This is question #{current_question_num}. "
            f"Generate a {difficulty} level technical question that:\n"
            f"1. Is different from any previous questions\n"
            f"2. Builds upon or relates to the candidate's previous answers if applicable\n"
            f"3. Is appropriate for a {session['user_role']} role\n"
            f"4. Tests {difficulty} level knowledge\n\n"
            f"{previous_qa_context}"
            f"Generate ONLY the next interview question, no additional text or formatting."
        )
        
        # Generate question with improved prompt
        question_text = generate_question(context)
        
        # Clean up the question text
        question_text = question_text.strip()
        if not question_text:
            question_text = f"Can you tell me about your experience with {session['user_role']} technologies?"
        
        # Store the generated question in session for future reference
        if "generated_questions" not in session:
            session["generated_questions"] = []
        
        session["generated_questions"].append({
            "question_number": current_question_num,
            "question": question_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate question audio
        question_audio_path = os.path.join(session_path, f"ai_question_{current_question_num}.wav")
        speak_text(question_text, question_audio_path)
        
        # Check if file was created
        if not os.path.exists(question_audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate question audio")
        
        # MONGODB INTEGRATION: Store conversation in database
        db_success = interview_manager.handle_ai_response(
            session_id=request.session_id,
            question_text=question_text,
            question_number=current_question_num,
            difficulty_level=difficulty,
            user_transcript=request.user_transcript,
            question_audio_path=question_audio_path
        )
        
        if not db_success:
            logger.warning(f"Failed to store AI response in database for session {request.session_id}")
        
        # Update session
        session["db_stored"] = db_success
        sessions[request.session_id] = session
        
        # Read audio file and encode as base64
        with open(question_audio_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Return JSON response with both text and audio
        return {
            "session_id": request.session_id,
            "question_number": current_question_num,
            "question_text": question_text,
            "audio_data": audio_data,
            "audio_format": "wav",
            "difficulty_level": difficulty,
            "message": "Question generated successfully",
            "db_stored": db_success
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate AI response: {str(e)}")

# Additional endpoints for MongoDB operations

@app.post("/api/session/{session_id}/analyze", response_model=SessionScoreResponse)
async def analyze_session_scores(session_id: str):
    """
    Analyze complete session and generate comprehensive scores for:
    - Confidence (1-100)
    - Technical Knowledge (1-100) 
    - Communication Skills (1-100)
    - Fluency (1-100)
    - Base Knowledge (1-100)
    
    Returns overall score and detailed feedback with strengths and improvements.
    """
    try:
        # Get session data from MongoDB
        session_data = interview_manager.interview_collection.get_interview_session(session_id)
        
        if not session_data:
            # Try to get from in-memory sessions as fallback
            if session_id in sessions:
                memory_session = sessions[session_id]
                # Convert in-memory session to expected format
                session_data = {
                    "session_id": session_id,
                    "user_name": memory_session.get("user_name"),
                    "user_role": memory_session.get("user_role"),
                    "conversation": [],
                    "created_at": memory_session.get("created_at")
                }
                
                # Add previous answers as conversation
                for answer in memory_session.get("previous_answers", []):
                    session_data["conversation"].append({
                        "type": "user_response",
                        "content": answer.get("answer", ""),  # Use 'content' field
                        "question_number": answer.get("question_number", 1)
                    })
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Debug: Log the session data structure
        print(f"Session data structure: {json.dumps(session_data, indent=2, default=str)}")
        
        # Ensure conversation exists and has proper format
        if "conversation" not in session_data:
            session_data["conversation"] = []
        
        # Convert conversation format if needed
        formatted_conversation = []
        for msg in session_data.get("conversation", []):
            if isinstance(msg, dict):
                # Handle different message formats
                if msg.get("type") == "user_response":
                    # Ensure content field exists
                    content = msg.get("content") or msg.get("transcript", "")
                    if content:
                        formatted_conversation.append({
                            "type": "user_response",
                            "content": content,
                            "timestamp": msg.get("timestamp"),
                            "question_number": msg.get("question_number")
                        })
                else:
                    # Keep other message types as-is
                    formatted_conversation.append(msg)
        
        session_data["conversation"] = formatted_conversation
        
        # Debug: Log user responses
        user_responses = [msg.get("content", "") for msg in formatted_conversation if msg.get("type") == "user_response"]
        print(f"Found {len(user_responses)} user responses: {user_responses}")
        
        if not user_responses:
            # Return default scores if no responses found
            return SessionScoreResponse(
                overallScore=50,
                scores={
                    "confidence": 50,
                    "technical": 50,
                    "communication": 50,
                    "fluency": 50,
                    "base_knowledge": 50
                },
                feedback={
                    "strengths": ["Participated in interview process"],
                    "improvements": ["No responses available for analysis"],
                    "detailedFeedback": "Unable to analyze session - no user responses found."
                },
                analysis_timestamp=datetime.now().isoformat()
            )
        
        # Analyze the session
        analysis_result = session_analyzer.analyze_session_scores(session_data)
        
        # Create combined sentiment analysis with all scores
        sentiment_analysis = {
            "confidence": analysis_result["scores"]["confidence"],
            "technical": analysis_result["scores"]["technical"],
            "communication": analysis_result["scores"]["communication"],
            "fluency": analysis_result["scores"]["fluency"],
            "base_knowledge": analysis_result["scores"]["base_knowledge"]
        }
        
        # Update the session with scores in MongoDB
        try:
            success = interview_manager.interview_collection.update_session_scores(
                session_id=session_id,
                sentiment_analysis=sentiment_analysis,
                confidence_score=analysis_result["scores"]["confidence"],
                communication_score=analysis_result["scores"]["communication"],
                technical_score=analysis_result["scores"]["technical"],
                overall_score=analysis_result["overallScore"],
                detailed_feedback=analysis_result["feedback"]
            )
            
            if not success:
                print(f"Warning: Failed to update scores in database for session {session_id}")
        except Exception as db_error:
            print(f"Database update error: {db_error}")
            # Continue without failing the API call
        
        return SessionScoreResponse(
            overallScore=analysis_result["overallScore"],
            scores=analysis_result["scores"],
            feedback=analysis_result["feedback"],
            analysis_timestamp=analysis_result["analysis_timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_session_scores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze session: {str(e)}")
@app.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session from MongoDB"""
    try:
        conversation = interview_manager.interview_collection.get_conversation_history(session_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Session not found or no conversation history")
        
        # Debug: Log the conversation structure
        print(f"Retrieved conversation with {len(conversation)} messages")
        user_responses = [msg for msg in conversation if msg.get("type") == "user_response"]
        print(f"Found {len(user_responses)} user responses")
        
        return {
            "session_id": session_id,
            "conversation": conversation,
            "total_messages": len(conversation)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")

@app.get("/session/{session_id}")
async def get_session_details(session_id: str):
    """Get complete session details from MongoDB"""
    try:
        session = interview_manager.interview_collection.get_interview_session(session_id)
        print(f"Retrieving session details for {session_id}")
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Debug: Log session structure
        if "conversation" in session:
            user_responses = [msg for msg in session["conversation"] if msg.get("type") == "user_response"]
            print(f"Session has {len(user_responses)} user responses")
        
        return session
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {str(e)}")

# Test endpoint to debug conversation format
@app.get("/debug/session/{session_id}")
async def debug_session_format(session_id: str):
    """Debug endpoint to check session data format"""
    try:
        session_data = interview_manager.interview_collection.get_interview_session(session_id)
        print(f"Debugging session {session_id} format")
        
        if not session_data:
            return {"error": "Session not found"}
        
        # Print the full session data structure to understand what we're working with
        print("=== FULL SESSION DATA ===")
        print(session_data)
        print("=== SESSION DATA KEYS ===")
        print(list(session_data.keys()) if isinstance(session_data, dict) else "Not a dict")
        
        # More flexible approach - check all possible conversation field names
        conversation_data = None
        conversation_field = None
        
        # Check common field names for conversation data
        possible_fields = ['conversation', 'messages', 'chat_history', 'interactions', 'responses']
        for field in possible_fields:
            if field in session_data and isinstance(session_data[field], list):
                conversation_data = session_data[field]
                conversation_field = field
                break
        
        # Extract user responses for debugging
        user_responses = []
        message_types = set()
        
        if conversation_data:
            for msg in conversation_data:
                # Collect all message types to understand structure
                if isinstance(msg, dict):
                    msg_type = msg.get("type", "unknown")
                    message_types.add(msg_type)
                    
                    # Look for user responses with flexible field matching
                    if msg_type == "user_response" or "user" in msg_type.lower():
                        user_responses.append({
                            "content": msg.get("content"),
                            "transcript": msg.get("transcript"),
                            "timestamp": msg.get("timestamp"),
                            "question_number": msg.get("question_number"),
                            "full_message": msg  # Include full message for debugging
                        })
        
        # Comprehensive session analysis
        return {
            "session_id": session_id,
            "session_found": True,
            "session_keys": list(session_data.keys()),
            "conversation_field_used": conversation_field,
            "total_messages": len(conversation_data) if conversation_data else 0,
            "user_responses_count": len(user_responses),
            "user_responses": user_responses,
            "session_structure": {
                "has_conversation": conversation_field is not None,
                "conversation_field": conversation_field,
                "conversation_length": len(conversation_data) if conversation_data else 0,
                "message_types": list(message_types),
                "all_fields": list(session_data.keys())
            },
            "raw_sample": {
                "first_few_items": dict(list(session_data.items())[:5]) if session_data else None
            }
        }
        
    except Exception as e:
        print(f"Debug endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Debug failed: {str(e)}"}

# Additional utility endpoint for testing scoring
@app.post("/test/analyze-text")
async def test_analyze_text(session_id: str, role: str = ""):
    """Test endpoint to analyze text directly without session"""
    try:
        responses = interview_manager.interview_collection.get_conversation_history(session_id)

        if not responses or not isinstance(responses, list):
            raise HTTPException(status_code=400, detail="No responses provided")

        # Construct mock session
        conversation = [
            {
                "type": message.get("type", ""),
                "content": message.get("content", ""),
                "question_number": message.get("metadata", {}).get("question_number", None),
                "speaker": message.get("speaker", "unknown"),
            }
            for message in responses
            if message.get("type") in ("user_response", "ai_question", "ai_greeting")
        ]

        if not conversation:
            raise HTTPException(status_code=400, detail="No valid conversation messages found")

        mock_session = {
            "session_id": session_id,
            "user_role": role or "Software Engineer",
            "conversation": conversation
        }

        analysis_result = session_analyzer.analyze_session_scores(mock_session)
        return analysis_result

    except HTTPException as e:
        raise e  # re-raise so correct status is preserved
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze text: {str(e)}")



@app.post("/api/sessions/compare")
async def compare_sessions(session_ids: List[str]):
    """
    Compare multiple interview sessions and provide comparative analysis
    """
    try:
        if len(session_ids) < 2 or len(session_ids) > 5:
            raise HTTPException(status_code=400, detail="Can compare 2-5 sessions at a time")
        
        comparisons = []
        
        for session_id in session_ids:
            try:
                session_data = interview_manager.interview_collection.get_interview_session(session_id)
                if session_data:
                    analysis = session_analyzer.analyze_session_scores(session_data)
                    comparisons.append({
                        "session_id": session_id,
                        "user_name": session_data.get("user_name", "Unknown"),
                        "user_role": session_data.get("user_role", "Unknown"),
                        "overall_score": analysis["overallScore"],
                        "scores": analysis["scores"],
                        "created_at": session_data.get("created_at")
                    })
            except:
                continue
        
        if len(comparisons) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid sessions to compare")
        
        # Calculate comparative metrics
        score_params = ["confidence", "technical", "communication", "fluency", "base_knowledge"]
        param_averages = {}
        for param in score_params:
            param_averages[param] = sum(comp["scores"][param] for comp in comparisons) / len(comparisons)
        
        overall_avg = sum(comp["overall_score"] for comp in comparisons) / len(comparisons)
        
        return {
            "session_comparisons": comparisons,
            "comparative_analysis": {
                "average_scores": param_averages,
                "overall_average": round(overall_avg, 1),
                "best_performer": max(comparisons, key=lambda x: x["overall_score"]),
                "most_improved_areas": [param for param in score_params if param_averages[param] >= 75],
                "needs_attention": [param for param in score_params if param_averages[param] < 60]
            },
            "comparison_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare sessions: {str(e)}")

# Bulk analysis endpoint for multiple sessions
@app.post("/api/sessions/bulk-analyze")
async def bulk_analyze_sessions(session_ids: List[str]):
    """
    Analyze multiple sessions in bulk and return aggregated insights
    """
    try:
        if len(session_ids) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 sessions can be analyzed at once")
        
        results = []
        failed_sessions = []
        
        for session_id in session_ids:
            try:
                session_data = interview_manager.interview_collection.get_interview_session(session_id)
                if session_data:
                    analysis = session_analyzer.analyze_session_scores(session_data)
                    results.append({
                        "session_id": session_id,
                        "analysis": analysis
                    })
                    
                    # Update database with scores
                    interview_manager.interview_collection.update_session_scores(
                        session_id=session_id,
                        sentiment_analysis=analysis["scores"],
                        confidence_score=analysis["scores"]["confidence"],
                        communication_score=analysis["scores"]["communication"],
                        technical_score=analysis["scores"]["technical"],
                        overall_score=analysis["overallScore"],
                        detailed_feedback=analysis["feedback"]
                    )
                else:
                    failed_sessions.append(session_id)
            except Exception as e:
                failed_sessions.append(f"{session_id}: {str(e)}")
        
        return {
            "successful_analyses": len(results),
            "failed_sessions": failed_sessions,
            "results": results,
            "bulk_analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform bulk analysis: {str(e)}")
@app.post("/session/{session_id}/finalize")
async def finalize_interview_session(session_id: str, audio_recording_url: str = None):
    """Finalize interview session"""
    try:
        success = interview_manager.interview_collection.finalize_session(
            session_id=session_id,
            audio_recording_url=audio_recording_url
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session finalized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to finalize session: {str(e)}")

# Cleanup function for graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection on app shutdown"""
    db_handler.close_connection()
@app.post("/analyze-answer", response_model=AnalysisResponse)
async def analyze_user_answer(
    session_id: str = Form(...),
    question_number: int = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Analyze user's audio answer - transcribe and extract features
    """
    try:
        # Get session data
        session = get_session(session_id)
        session_path = session["session_path"]
        
        # Save uploaded audio file
        audio_filename = f"user_answer_{question_number}.wav"
        audio_path = os.path.join(session_path, audio_filename)
        
        with open(audio_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Transcribe audio
        transcript = ""
        try:
            transcript = transcribe_audio(audio_path)
            # Save transcript
            with open(os.path.join(session_path, f"transcript_{question_number}.txt"), "w") as f:
                f.write(transcript)
        except Exception as e:
            print(f"Transcription failed: {e}")
        
        # Extract voice features and analyze emotion
        nervousness_score = None
        features_dict = None
        try:
            features = extract_voice_features(audio_path)
            rel_features = compute_relative_features(features)
            nervousness_score = score_nervousness_relative(rel_features)
            features_dict = {
                "raw_features": features,
                "relative_features": rel_features
            }
        except Exception as e:
            print(f"Feature extraction failed: {e}")
        
        return AnalysisResponse(
            transcript=transcript,
            nervousness_score=nervousness_score,
            features=features_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze answer: {str(e)}")

@app.get("/audio/{session_id}/{filename}")
async def get_audio_file(session_id: str, filename: str):
    """
    Serve audio files for playback
    """
    try:
        session = get_session(session_id)
        audio_path = os.path.join(session["session_path"], filename)
        
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve audio: {str(e)}")

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """
    Get session information
    """
    try:
        session = get_session(session_id)
        return {
            "session_id": session_id,
            "user_name": session["user_name"],
            "user_role": session["user_role"],
            "current_question": session["current_question"],
            "total_answers": len(session["previous_answers"]),
            "created_at": session["created_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """
    End interview session and cleanup
    """
    try:
        if session_id in sessions:
            del sessions[session_id]
        return {"message": "Session ended successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")

@app.post("/generate-assessment")
async def generate_assessment(
    id: str = None,
):
    """
    Generate a custom assessment based on user profile data
    
    Args:
        id: User ID to fetch profile information
        
    Returns:
        Generated assessment with rounds and configuration
    """
    async with httpx.AsyncClient() as client:
        try:
            # Fetch user profile data
            response = await client.get(f"http://localhost:5000/api/user/analysis/details/{id}")
            response.raise_for_status()
            
            user_profile = response.json()
            print(f"Received user profile: {user_profile}")
            
            # Create UserProfileRequest instance
            user = UserProfileRequest(**user_profile)
            print(f"Validated user profile: {user}")
            
            user_data = user.model_dump()
            print(f"User data for assessment generation: {user_data}")
            
            # Initialize the assessment generator
            generator = AssessmentGenerator(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Generate the assessment
            assessment_data = generator.generate_assessment(user_data)
            
            # Add userId field to assessment_data
            assessment_data["userId"] = id
            
            #asses_data = AssessmentResponse(**assessment_data)
            print("Assessment data validated successfully",assessment_data)
            
            # Send the assessment data with userId to the final API
            res = await client.post(
                "http://localhost:5000/api/custom-assessments",
                json=assessment_data
            )
            res.raise_for_status()
            print(f"Assessment successfully created: {res.json()}")
            
            return res.json()
                    
        except httpx.HTTPError as he:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to fetch user data: {str(he)}"
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate assessment: {str(e)}"
            )
@app.post("/generate-assessment-raw")
async def generate_assessment_raw(
    user_profile: UserProfileRequest,
    api_key: str = os.getenv("GEMINI_API_KEY")
):
    """
    Generate a custom assessment and return raw JSON response
    
    Args:
        user_profile: User profile information including skills, target role, etc.
    
    Returns:
        Raw JSON response with generated assessment
    """
    try:
        # Convert Pydantic model to dictionary
        user_data = user_profile.model_dump()
        
        # Initialize the assessment generator
        generator = AssessmentGenerator(api_key)
        
        # Generate the assessment
        assessment_data = generator.generate_assessment(user_data)
        
        return JSONResponse(content=assessment_data)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate assessment: {str(e)}"
        )

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