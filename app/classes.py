from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

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

class TechnicalGenerationInput(BaseModel):
    roundType: str
    difficulty: str
    questionCount: int
    category: Optional[str]
    duration: int
    type: str

class GenerateTechnicalQuestionsRequest(BaseModel):
    questions_with_answers: List[dict]

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

class Question(BaseModel):
    question: str
    answer: str
class QuestionApt(BaseModel):
    question: str
    answer: str

class EvaluationRequest(BaseModel):
    questions: List[QuestionApt]

class EvaluationResponse(BaseModel):
    overallScore: str
    feedback: Dict[str, Any]
    detailedResults: List[Dict[str, Any]]

class EvaluationRequestTechnical(BaseModel):
    questions: List[Question]
    question_type: Literal["aptitude", "technical", "os", "cn", "dbms"] = "aptitude"
class GenerateAptitudeQuestionsRequest(BaseModel):
    questions_with_answers: List[dict]
# Communication Question Generator Class
class GenerateTechnicalQuestionsRequest(BaseModel):
    questions_with_answers: List[dict]

class GreetingRequest(BaseModel):
    user_name: str
    user_role: str
    round_id: str = None

class AIResponseRequest(BaseModel):
    session_id: str
    question_number: Optional[int] = 1
    user_transcript: Optional[str] = None

class AnalysisResponse(BaseModel):
    transcript: str
    nervousness_score: Optional[float] = None
    features: Optional[dict] = None

class SessionScoreResponse(BaseModel):
    overallScore: int
    scores: Dict[str, int]
    feedback: Dict[str, Any]
    analysis_timestamp: str

class SessionAnalysisRequest(BaseModel):
    session_id: str

class UserProfileRequest(BaseModel):
    userId: str = Field(..., description="Unique user identifier")
    skills: List[str] = Field(default=[], description="List of user skills")
    contributionFreq: str = Field(default="medium", description="Contribution frequency")
    projectsCount: int = Field(default=0, description="Number of projects")
    topLanguages: List[str] = Field(default=[], description="Top programming languages")
    recentActivity: Dict[str, Any] = Field(default={}, description="Recent activity data")
    repositoryStats: Dict[str, Any] = Field(default={}, description="Repository statistics")
    targetRole: str = Field(..., description="Target job role")
    dreamCompanies: List[str] = Field(default=[], description="Dream companies list")
    skillGaps: List[str] = Field(default=[], description="Identified skill gaps")
    careerPath: List[str] = Field(default=[], description="Career path progression")

class RoundResponse(BaseModel):
    roundType: str
    name: str
    description: Optional[str]
    duration: int
    sequence: int
    config: Optional[Dict[str, Any]] = None

class AssessmentResponse(BaseModel):
    name: str
    description: Optional[str]
    difficulty: str
    rounds: List[RoundResponse]

class RoundType(Enum):
    SCREENING = 'SCREENING'
    APTITUDE = 'APTITUDE'
    COMMUNICATION = 'COMMUNICATION'
    CODING = 'CODING'
    TECHNICAL = 'TECHNICAL'
    BEHAVIORAL = 'BEHAVIORAL'
    SYSTEM_DESIGN = 'SYSTEM_DESIGN'

class DifficultyLevel(Enum):
    EASY = 'EASY'
    MEDIUM = 'MEDIUM'
    HARD = 'HARD'

@dataclass
class CreateCustomRoundDto:
    roundType: str
    name: str
    description: Optional[str]
    duration: int  # in minutes
    sequence: int
    config: Optional[Dict[str, Any]] = None

@dataclass
class CreateCustomAssessmentDto:
    name: str
    description: Optional[str]
    difficulty: str
    rounds: List[CreateCustomRoundDto]
