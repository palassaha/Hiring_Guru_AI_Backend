# MongoDB Implementation for InterviewData Model
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError
from datetime import datetime
from bson import ObjectId
import logging
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBHandler:
    def __init__(self, connection_string=DATABASE_URL, database_name="h4b-2025"):
        """
        Initialize MongoDB connection for Interview Data
        
        Args:
            connection_string (str): MongoDB connection URI
            database_name (str): Name of the database
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                maxPoolSize=50,
                retryWrites=True
            )
            
            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"Successfully connected to MongoDB database: {self.database_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

class InterviewDataCollection:
    """Handle InterviewData collection operations"""
    
    def __init__(self, db_handler):
        self.db = db_handler.db
        self.collection = self.db.interview_data
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for InterviewData collection"""
        try:
            # Create unique indexes
            self.collection.create_index("sessionId", unique=True)
            self.collection.create_index("roundId", unique=True)
            
            # Create compound indexes for queries
            self.collection.create_index([("sessionId", 1), ("createdAt", -1)])
            self.collection.create_index([("roundId", 1), ("createdAt", -1)])
            
            logger.info("InterviewData indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def create_interview_session(self, session_id: str, round_id: str, auth_token: str, 
                               model_backend: str = "openai-gpt", user_name: str = "", 
                               user_role: str = "") -> Optional[str]:
        """
        Create a new interview session record
        
        Args:
            session_id: Unique session identifier
            round_id: Round identifier this interview belongs to
            auth_token: Authentication token for backend model connection
            model_backend: AI model being used
            user_name: Name of the interviewee
            user_role: Role being interviewed for
            
        Returns:
            ObjectId string if successful, None if failed
        """
        try:
            # Initialize transcript with session start
            initial_transcript = {
                "session_start": datetime.now().isoformat(),
                "user_name": user_name,
                "user_role": user_role,
                "conversation": [],
                "session_metadata": {
                    "total_questions": 0,
                    "total_responses": 0,
                    "session_duration": None
                }
            }
            
            interview_data = {
                "roundId": round_id,
                "sessionId": session_id,
                "authToken": auth_token,
                "transcript": initial_transcript,
                "audioRecording": None,
                "sentimentAnalysis": None,
                "confidenceScore": None,
                "communicationScore": None,
                "technicalScore": None,
                "modelBackend": model_backend,
                "connectionLogs": {
                    "session_created": datetime.now().isoformat(),
                    "connections": [],
                    "disconnections": []
                },
                "createdAt": datetime.now(),
                "updatedAt": datetime.now()
            }
            
            result = self.collection.insert_one(interview_data)
            logger.info(f"Interview session created with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except DuplicateKeyError as e:
            logger.error(f"Session ID or Round ID already exists: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating interview session: {e}")
            return None
    
    def add_greeting_to_transcript(self, session_id: str, greeting_text: str, 
                                 audio_path: str = None) -> bool:
        """
        Add greeting to the conversation transcript
        
        Args:
            session_id: Session identifier
            greeting_text: The greeting message
            audio_path: Path to greeting audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            greeting_entry = {
                "type": "ai_greeting",
                "timestamp": datetime.now().isoformat(),
                "speaker": "ai",
                "content": greeting_text,
                "audio_file": audio_path,
                "metadata": {
                    "message_type": "greeting",
                    "question_number": 0
                }
            }
            
            result = self.collection.update_one(
                {"sessionId": session_id},
                {
                    "$push": {"transcript.conversation": greeting_entry},
                    "$set": {"updatedAt": datetime.now()}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Greeting added to session {session_id}")
                return True
            else:
                logger.warning(f"No session found with ID {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding greeting to transcript: {e}")
            return False
    
    def add_ai_question_to_transcript(self, session_id: str, question_text: str, 
                                    question_number: int, difficulty_level: str = "basic",
                                    audio_path: str = None) -> bool:
        """
        Add AI question to the conversation transcript
        
        Args:
            session_id: Session identifier
            question_text: The question text
            question_number: Question sequence number
            difficulty_level: Difficulty level of the question
            audio_path: Path to question audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            question_entry = {
                "type": "ai_question",
                "timestamp": datetime.now().isoformat(),
                "speaker": "ai",
                "content": question_text,
                "audio_file": audio_path,
                "metadata": {
                    "question_number": question_number,
                    "difficulty_level": difficulty_level,
                    "message_type": "question"
                }
            }
            
            result = self.collection.update_one(
                {"sessionId": session_id},
                {
                    "$push": {"transcript.conversation": question_entry},
                    "$inc": {"transcript.session_metadata.total_questions": 1},
                    "$set": {"updatedAt": datetime.now()}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"AI question {question_number} added to session {session_id}")
                return True
            else:
                logger.warning(f"No session found with ID {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding AI question to transcript: {e}")
            return False
    
    def add_user_response_to_transcript(self, session_id: str, user_transcript: str, 
                                      question_number: int, audio_path: str = None) -> bool:
        """
        Add user response to the conversation transcript
        
        Args:
            session_id: Session identifier
            user_transcript: User's response text
            question_number: Question number this response is for
            audio_path: Path to user's audio response
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response_entry = {
                "type": "user_response",
                "timestamp": datetime.now().isoformat(),
                "speaker": "user",
                "content": user_transcript,
                "audio_file": audio_path,
                "metadata": {
                    "question_number": question_number,
                    "message_type": "response",
                    "response_length": len(user_transcript.split())
                }
            }
            
            result = self.collection.update_one(
                {"sessionId": session_id},
                {
                    "$push": {"transcript.conversation": response_entry},
                    "$inc": {"transcript.session_metadata.total_responses": 1},
                    "$set": {"updatedAt": datetime.now()}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"User response to question {question_number} added to session {session_id}")
                return True
            else:
                logger.warning(f"No session found with ID {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding user response to transcript: {e}")
            return False
    
    def get_interview_session(self, session_id: str) -> Optional[Dict]:
        """Get interview session by session ID"""
        try:
            print(f"Searching for session: {session_id}")
            session = self.collection.find_one({"sessionId": session_id})
            print(f"Query executed. Session found: {session is not None}")
            
            if session:
                print(f"Session keys: {list(session.keys())}")
                print(f"Session data preview: {str(session)[:200]}...")
                # Convert ObjectId to string for JSON serialization
                session["_id"] = str(session["_id"])
                return session
            else:
                print("No session found with that ID")
                # Let's also check if there are any sessions at all
                total_sessions = self.collection.count_documents({})
                print(f"Total sessions in collection: {total_sessions}")
                
                # Check for similar session IDs
                similar_sessions = list(self.collection.find({}, {"sessionId": 1}).limit(5))
                print(f"Sample session IDs in database: {[s.get('sessionId') for s in similar_sessions]}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving session: {e}")
            print(f"Exception in get_interview_session: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        try:
            session = self.collection.find_one(
                {"sessionId": session_id},
                {"transcript.conversation": 1}
            )
            
            if session and "transcript" in session:
                return session["transcript"].get("conversation", [])
            return []
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def update_session_scores(self, session_id: str, sentiment_analysis: Dict = None,
                            confidence_score: float = None, communication_score: float = None,
                            technical_score: float = None) -> bool:
        """Update analysis scores for a session"""
        try:
            update_data = {"updatedAt": datetime.now()}
            
            if sentiment_analysis:
                update_data["sentimentAnalysis"] = sentiment_analysis
            if confidence_score is not None:
                update_data["confidenceScore"] = confidence_score
            if communication_score is not None:
                update_data["communicationScore"] = communication_score
            if technical_score is not None:
                update_data["technicalScore"] = technical_score
            
            result = self.collection.update_one(
                {"sessionId": session_id},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating session scores: {e}")
            return False
    
    def finalize_session(self, session_id: str, audio_recording_url: str = None) -> bool:
        """Mark session as complete and update final metadata"""
        try:
            # Get session to calculate duration
            session = self.get_interview_session(session_id)
            if not session:
                return False
            
            # Calculate session duration
            created_at = session.get("createdAt")
            if created_at:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                duration = (datetime.now() - created_at).total_seconds()
            else:
                duration = None
            
            update_data = {
                "transcript.session_metadata.session_duration": duration,
                "transcript.session_end": datetime.now().isoformat(),
                "updatedAt": datetime.now()
            }
            
            if audio_recording_url:
                update_data["audioRecording"] = audio_recording_url
            
            result = self.collection.update_one(
                {"sessionId": session_id},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error finalizing session: {e}")
            return False

# Integration with your FastAPI functions
class InterviewDataManager:
    """Manager class to integrate with your FastAPI endpoints"""
    
    def __init__(self, db_handler):
        self.interview_collection = InterviewDataCollection(db_handler)
    
    def handle_greeting_creation(self, session_id: str, user_name: str, user_role: str, 
                               greeting_text: str, round_id: str, auth_token: str,
                               greeting_audio_path: str = None) -> bool:
        """
        Handle the greeting creation from your /greeting endpoint
        """
        try:
            # Create interview session
            interview_id = self.interview_collection.create_interview_session(
                session_id=session_id,
                round_id=round_id,
                auth_token=auth_token,
                model_backend="openai-gpt",
                user_name=user_name,
                user_role=user_role
            )
            
            if not interview_id:
                return False
            
            # Add greeting to transcript
            return self.interview_collection.add_greeting_to_transcript(
                session_id=session_id,
                greeting_text=greeting_text,
                audio_path=greeting_audio_path
            )
            
        except Exception as e:
            logger.error(f"Error handling greeting creation: {e}")
            return False
    
    def handle_ai_response(self, session_id: str, question_text: str, question_number: int,
                         difficulty_level: str, user_transcript: str = None,
                         question_audio_path: str = None) -> bool:
        """
        Handle AI response from your /ai-response endpoint
        """
        try:
            success = True
            
            # Add user response if provided
            if user_transcript and user_transcript.strip():
                success = self.interview_collection.add_user_response_to_transcript(
                    session_id=session_id,
                    user_transcript=user_transcript,
                    question_number=question_number - 1  # Previous question response
                )
            
            # Add AI question
            if success:
                success = self.interview_collection.add_ai_question_to_transcript(
                    session_id=session_id,
                    question_text=question_text,
                    question_number=question_number,
                    difficulty_level=difficulty_level,
                    audio_path=question_audio_path
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling AI response: {e}")
            return False
    

# Usage Example
