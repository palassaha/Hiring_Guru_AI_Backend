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
        """Initialize the SessionAnalyzer with Groq API key"""
        self.client = openai
        self.client.api_key = groq_api_key
        self.client.api_base = "https://api.groq.com/openai/v1"
    
    def analyze_session_scores(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a complete interview session and generate comprehensive scores"""
        try:
            print(f"=== SESSION DATA ANALYSIS START ===")
            print(f"Session ID: {session_data.get('session_id', 'Unknown')}")
            
            conversation_history = self._extract_conversation_from_session(session_data)
            user_responses = self._extract_user_responses(conversation_history)
            
            print(f"Found {len(user_responses)} user responses")
            if not user_responses:
                return self._generate_default_scores("No user responses found")
            
            user_role = self._extract_user_role(session_data)
            print(f"User role: {user_role}")
            
            scores = {
                "confidence": self._analyze_confidence(user_responses),
                "technical": self._analyze_technical_knowledge(user_responses, user_role),
                "communication": self._analyze_communication(user_responses),
                "fluency": self._analyze_fluency(user_responses),
                "base_knowledge": self._analyze_base_knowledge(user_responses, user_role)
            }
            
            print(f"Generated scores: {scores}")
            overall_score = self._calculate_overall_score(scores)
            feedback = self._generate_comprehensive_feedback(scores, user_responses, user_role)
            
            print(f"=== SESSION DATA ANALYSIS END ===")
            return {
                "overallScore": overall_score,
                "scores": scores,
                "feedback": feedback,
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error analyzing session: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_default_scores(f"Analysis error: {str(e)}")

    def _extract_conversation_from_session(self, session_data: Dict[str, Any]) -> List[Dict]:
        """Extract conversation messages"""
        conversation = session_data.get("conversation", [])
        print(f"Found conversation with {len(conversation)} messages")
        return conversation

    def _extract_user_role(self, session_data: Dict[str, Any]) -> str:
        """Extract the user role"""
        return session_data.get("user_role", "Software Engineer")

    def _extract_user_responses(self, conversation: List[Dict]) -> List[str]:
        """Extract user responses only"""
        responses = []
        print(f"Processing {len(conversation)} conversation messages")
        for i, message in enumerate(conversation):
            print(f"Message {i}: type={message.get('type')}, speaker={message.get('speaker')}")
            if message.get("type") == "user_response":
                response_text = message.get("content", "").strip()
                if response_text:
                    responses.append(response_text)
                    print(f"Added response: '{response_text[:50]}...'")
        print(f"Extracted {len(responses)} total responses")
        return responses

    def _analyze_confidence(self, responses: List[str]) -> int:
        """Analyze confidence heuristically and with LLM"""
        try:
            if not responses:
                return 50

            all_responses = '\n'.join(responses)
            confidence_indicators = [
                "I am confident", "I believe", "definitely", "certainly",
                "clearly", "obviously", "sure", "absolutely"
            ]
            hesitation_indicators = [
                "I think", "maybe", "probably", "I'm not sure", "perhaps",
                "I guess", "kind of", "sort of", "possibly"
            ]

            confidence_count = sum(1 for word in confidence_indicators
                                   if any(word.lower() in r.lower() for r in responses))
            hesitation_count = sum(1 for word in hesitation_indicators
                                   if any(word.lower() in r.lower() for r in responses))

            total_words = sum(len(r.split()) for r in responses)
            avg_len = total_words / len(responses)

            base_score = 50
            if confidence_count > hesitation_count:
                base_score += 20
            elif hesitation_count > confidence_count:
                base_score -= 15

            if avg_len > 30:
                base_score += 10
            elif avg_len < 10:
                base_score -= 10

            try:
                prompt = f"""
                Analyze confidence level (1-100) based on these interview responses:
                {all_responses[:1000]}...
                Respond with only a number 1-100.
                """
                response = self.client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=10
                )
                llm_score = self._extract_score_from_text(response.choices[0].message.content.strip())
                return max(1, min(100, (base_score + llm_score) // 2))

            except Exception as e:
                print(f"LLM fallback failed: {e}")
                return base_score

        except Exception as e:
            print(f"Confidence analysis error: {e}")
            return 75

    def _analyze_technical_knowledge(self, responses: List[str], role: str) -> int:
        """Analyze technical depth"""
        try:
            all_text = '\n'.join(responses)
            keywords = [
                'algorithm', 'data structure', 'database', 'API', 'framework',
                'programming', 'coding', 'software', 'system', 'architecture',
                'class', 'method', 'function', 'variable', 'object', 'thread',
                'exception', 'null', 'synchronized', 'lock', 'deadlock',
                'hashmap', 'java', 'python', 'javascript', 'sql'
            ]
            keyword_count = sum(1 for k in keywords if any(k in r.lower() for r in responses))
            base_score = min(85, 40 + keyword_count * 3)

            try:
                prompt = f"""
                Rate technical knowledge (1-100) for {role} role based on responses:
                {all_text[:1000]}...
                Respond with only a number 1-100.
                """
                response = self.client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=10
                )
                llm_score = self._extract_score_from_text(response.choices[0].message.content.strip())
                return max(1, min(100, (base_score + llm_score) // 2))

            except Exception:
                return base_score

        except Exception as e:
            print(f"Technical knowledge error: {e}")
            return 70

    def _analyze_communication(self, responses: List[str]) -> int:
        """Analyze communication clarity and structure"""
        try:
            total_words = sum(len(r.split()) for r in responses)
            avg_len = total_words / len(responses)
            base_score = 50

            if avg_len > 25:
                base_score += 15
            elif avg_len < 8:
                base_score -= 15

            complete = sum(1 for r in responses if r.endswith('.') or len(r.split()) > 5)
            if complete == len(responses):
                base_score += 10

            return max(1, min(100, base_score))
        except Exception as e:
            print(f"Communication analysis error: {e}")
            return 75

    def _analyze_fluency(self, responses: List[str]) -> int:
        """Analyze fluency and filler usage"""
        try:
            text = ' '.join(responses).lower()
            fillers = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
            filler_count = sum(text.count(f) for f in fillers)
            total_words = len(text.split())

            base_score = 80
            if total_words > 0:
                ratio = filler_count / total_words
                if ratio > 0.05:
                    base_score -= 20
                elif ratio > 0.02:
                    base_score -= 10

            return max(1, min(100, base_score))
        except Exception as e:
            print(f"Fluency analysis error: {e}")
            return 80

    def _analyze_base_knowledge(self, responses: List[str], role: str) -> int:
        """Evaluate fundamental understanding"""
        try:
            indicators = [
                'understand', 'concept', 'principle', 'approach', 'method',
                'solution', 'problem', 'analysis', 'design', 'implementation'
            ]
            score = 50 + min(30, sum(1 for k in indicators if any(k in r.lower() for r in responses)) * 2)
            return max(1, min(100, score))
        except Exception as e:
            print(f"Base knowledge analysis error: {e}")
            return 70

    def _extract_score_from_text(self, text: str) -> int:
        """Extract numeric score from string"""
        import re
        match = re.findall(r'\d+', text)
        return int(match[0]) if match else 50

    def _calculate_overall_score(self, scores: Dict[str, int]) -> int:
        weights = {
            "confidence": 0.15,
            "technical": 0.30,
            "communication": 0.25,
            "fluency": 0.15,
            "base_knowledge": 0.15
        }
        return round(sum(scores[k] * weights[k] for k in weights))

    def _generate_comprehensive_feedback(self, scores: Dict[str, int], responses: List[str], role: str) -> Dict[str, Any]:
        strengths = []
        improvements = []

        if scores["confidence"] >= 75:
            strengths.append("Demonstrates strong confidence")
        if scores["technical"] >= 75:
            strengths.append("Solid technical understanding")
        if scores["communication"] >= 75:
            strengths.append("Clear and structured communicator")
        if scores["fluency"] >= 75:
            strengths.append("Fluent language and delivery")
        if scores["base_knowledge"] >= 75:
            strengths.append("Good foundational understanding")

        if scores["confidence"] < 60:
            improvements.append("Improve assertiveness and confidence")
        if scores["technical"] < 60:
            improvements.append("Strengthen core technical concepts")
        if scores["communication"] < 60:
            improvements.append("Structure responses more clearly")
        if scores["fluency"] < 60:
            improvements.append("Reduce fillers and improve flow")
        if scores["base_knowledge"] < 60:
            improvements.append("Study fundamental concepts for the role")

        if not strengths:
            strengths = ["Participated actively in the interview"]
        if not improvements:
            improvements = ["Maintain consistency and continue practicing"]

        avg_score = sum(scores.values()) / len(scores)
        if avg_score >= 80:
            summary = "Excellent performance with strong responses across all dimensions."
        elif avg_score >= 60:
            summary = "Good performance with potential to excel further with targeted improvements."
        else:
            summary = "Needs improvement in key areas for interview success."

        return {
            "strengths": strengths,
            "improvements": improvements,
            "detailedFeedback": summary
        }

    def _generate_default_scores(self, reason: str) -> Dict[str, Any]:
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
