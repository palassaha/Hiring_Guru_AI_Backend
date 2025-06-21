# session_analyzer.py
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

class SessionAnalyzer:
    def __init__(self, groq_api_key: str):
        """
        Initialize the SessionAnalyzer with Groq API key
        """
        self.client = openai
        self.client.api_key = groq_api_key
        self.client.api_base = "https://api.groq.com/openai/v1"
    
    def analyze_session_scores(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a complete interview session and generate comprehensive scores
        """
        try:
            # Extract conversation data
            conversation_history = session_data.get("conversation", [])
            user_responses = self._extract_user_responses(conversation_history)
            
            if not user_responses:
                return self._generate_default_scores("No user responses found")
            
            # Analyze each parameter
            scores = {
                "confidence": self._analyze_confidence(user_responses),
                "technical": self._analyze_technical_knowledge(user_responses, session_data.get("user_role", "Software Engineer")),
                "communication": self._analyze_communication(user_responses),
                "fluency": self._analyze_fluency(user_responses),
                "base_knowledge": self._analyze_base_knowledge(user_responses, session_data.get("user_role", "Software Engineer"))
            }
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(scores)
            
            # Generate feedback
            feedback = self._generate_comprehensive_feedback(scores, user_responses, session_data.get("user_role", "Software Engineer"))
            
            return {
                "overallScore": overall_score,
                "scores": scores,
                "feedback": feedback,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing session: {e}")
            return self._generate_default_scores(f"Analysis error: {str(e)}")
    
    def _extract_user_responses(self, conversation: List[Dict]) -> List[str]:
        """Extract user responses from conversation history"""
        responses = []
        for message in conversation:
            if message.get("type") == "user_response" and message.get("transcript"):
                responses.append(message["transcript"])
        return responses
    
    def _analyze_confidence(self, responses: List[str]) -> int:
        """Analyze confidence level based on speech patterns and content"""
        try:
            prompt = f"""
            Analyze the confidence level of a candidate based on their interview responses.
            
            Responses: {' | '.join(responses)}
            
            Rate confidence from 1-100 based on:
            - Use of assertive language vs hesitant phrases
            - Clarity and decisiveness in answers
            - Self-assurance indicators
            - Frequency of filler words or uncertainty markers
            
            Return only a number between 1-100.
            """
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            
            score = int(response.choices[0].message.content.strip())
            return max(1, min(100, score))
            
        except:
            return 75  # Default score
    
    def _analyze_technical_knowledge(self, responses: List[str], role: str) -> int:
        """Analyze technical knowledge and expertise"""
        try:
            prompt = f"""
            Analyze the technical knowledge of a {role} candidate based on their responses.
            
            Responses: {' | '.join(responses)}
            Role: {role}
            
            Rate technical knowledge from 1-100 based on:
            - Depth of technical understanding
            - Use of appropriate technical terminology
            - Problem-solving approach
            - Knowledge of relevant technologies and concepts
            - Ability to explain complex topics
            
            Return only a number between 1-100.
            """
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            
            score = int(response.choices[0].message.content.strip())
            return max(1, min(100, score))
            
        except:
            return 70  # Default score
    
    def _analyze_communication(self, responses: List[str]) -> int:
        """Analyze communication skills"""
        try:
            prompt = f"""
            Analyze the communication skills of a candidate based on their responses.
            
            Responses: {' | '.join(responses)}
            
            Rate communication skills from 1-100 based on:
            - Clarity and coherence of responses
            - Structure and organization of thoughts
            - Ability to articulate ideas effectively
            - Listening and responding appropriately
            - Professional communication style
            
            Return only a number between 1-100.
            """
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            
            score = int(response.choices[0].message.content.strip())
            return max(1, min(100, score))
            
        except:
            return 75  # Default score
    
    def _analyze_fluency(self, responses: List[str]) -> int:
        """Analyze language fluency and flow"""
        try:
            prompt = f"""
            Analyze the language fluency of a candidate based on their responses.
            
            Responses: {' | '.join(responses)}
            
            Rate fluency from 1-100 based on:
            - Smooth flow of speech (minimal hesitations)
            - Proper grammar and sentence structure
            - Vocabulary range and appropriateness
            - Natural rhythm and pace
            - Minimal repetition or filler words
            
            Return only a number between 1-100.
            """
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            
            score = int(response.choices[0].message.content.strip())
            return max(1, min(100, score))
            
        except:
            return 80  # Default score
    
    def _analyze_base_knowledge(self, responses: List[str], role: str) -> int:
        """Analyze fundamental knowledge relevant to the role"""
        try:
            prompt = f"""
            Analyze the fundamental knowledge of a {role} candidate.
            
            Responses: {' | '.join(responses)}
            Role: {role}
            
            Rate base knowledge from 1-100 based on:
            - Understanding of core concepts
            - Industry awareness and trends
            - Foundational skills and principles
            - General knowledge relevant to the role
            - Learning aptitude and curiosity
            
            Return only a number between 1-100.
            """
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            
            score = int(response.choices[0].message.content.strip())
            return max(1, min(100, score))
            
        except:
            return 70  # Default score
    
    def _calculate_overall_score(self, scores: Dict[str, int]) -> int:
        """Calculate weighted overall score"""
        weights = {
            "confidence": 0.15,
            "technical": 0.30,
            "communication": 0.25,
            "fluency": 0.15,
            "base_knowledge": 0.15
        }
        
        overall = sum(scores[param] * weights[param] for param in weights)
        return round(overall)
    
    def _generate_comprehensive_feedback(self, scores: Dict[str, int], responses: List[str], role: str) -> Dict[str, Any]:
        """Generate detailed feedback based on scores and responses"""
        try:
            prompt = f"""
            Generate comprehensive interview feedback for a {role} candidate.
            
            Scores:
            - Confidence: {scores['confidence']}/100
            - Technical Knowledge: {scores['technical']}/100
            - Communication: {scores['communication']}/100
            - Fluency: {scores['fluency']}/100
            - Base Knowledge: {scores['base_knowledge']}/100
            
            Sample Responses: {' | '.join(responses[:3])}
            
            Provide feedback in this exact JSON format:
            {{
                "strengths": ["strength1", "strength2", "strength3"],
                "improvements": ["improvement1", "improvement2", "improvement3"],
                "detailedFeedback": "Comprehensive paragraph about overall performance, specific observations, and actionable recommendations."
            }}
            
            Make the feedback specific, actionable, and professional.
            """
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=500
            )
            
            feedback_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON, fallback to structured format
            try:
                return json.loads(feedback_text)
            except:
                return self._parse_feedback_text(feedback_text, scores)
                
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return self._generate_default_feedback(scores)
    
    def _parse_feedback_text(self, text: str, scores: Dict[str, int]) -> Dict[str, Any]:
        """Parse feedback text if JSON parsing fails"""
        strengths = []
        improvements = []
        detailed_feedback = text
        
        # Extract strengths and improvements based on scores
        if scores['confidence'] >= 80:
            strengths.append("Shows strong confidence in responses")
        elif scores['confidence'] < 60:
            improvements.append("Work on building confidence and assertiveness")
            
        if scores['technical'] >= 80:
            strengths.append("Demonstrates solid technical knowledge")
        elif scores['technical'] < 60:
            improvements.append("Enhance technical skills and knowledge base")
            
        if scores['communication'] >= 80:
            strengths.append("Excellent communication and articulation skills")
        elif scores['communication'] < 60:
            improvements.append("Improve communication clarity and organization")
        
        return {
            "strengths": strengths or ["Shows potential in interview responses"],
            "improvements": improvements or ["Continue practicing interview skills"],
            "detailedFeedback": detailed_feedback or "Overall performance shows room for growth with continued practice."
        }
    
    def _generate_default_feedback(self, scores: Dict[str, int]) -> Dict[str, Any]:
        """Generate default feedback based on scores"""
        avg_score = sum(scores.values()) / len(scores)
        
        if avg_score >= 80:
            return {
                "strengths": ["Strong overall performance", "Good technical understanding", "Clear communication"],
                "improvements": ["Continue refining skills", "Practice complex scenarios"],
                "detailedFeedback": "Excellent interview performance with strong technical and communication skills. Continue building on this foundation."
            }
        elif avg_score >= 60:
            return {
                "strengths": ["Shows potential", "Basic understanding demonstrated", "Willing to engage"],
                "improvements": ["Enhance technical knowledge", "Improve communication clarity", "Build confidence"],
                "detailedFeedback": "Good foundation with room for improvement. Focus on strengthening technical skills and communication clarity."
            }
        else:
            return {
                "strengths": ["Shows willingness to learn", "Participates in discussion"],
                "improvements": ["Significant technical skill development needed", "Improve communication skills", "Build confidence"],
                "detailedFeedback": "Requires focused preparation and skill development. Recommend additional practice and study before next interview."
            }
    
    def _generate_default_scores(self, reason: str) -> Dict[str, Any]:
        """Generate default scores when analysis fails"""
        return {
            "overallScore": 50,
            "scores": {
                "confidence": 50,
                "technical": 50,
                "communication": 50,
                "fluency": 50,
                "base_knowledge": 50
            },
            "feedback": {
                "strengths": ["Participated in interview process"],
                "improvements": ["Unable to analyze due to insufficient data"],
                "detailedFeedback": f"Analysis could not be completed: {reason}"
            },
            "analysis_timestamp": datetime.now().isoformat()
        }