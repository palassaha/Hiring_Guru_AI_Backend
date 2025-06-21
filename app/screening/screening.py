from datetime import datetime
import json
import os
from typing import Any, List
from dotenv import load_dotenv
import openai
from sympy import Dict, re
load_dotenv()

# Configure OpenAI to use Groq's endpoint
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

class JobScreeningSystem:
    def __init__(self):
        """Initialize the screening system with OpenAI configuration"""
        if not openai.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set!")

    def parse_company_role(self, company_with_role: str) -> tuple:
        """
        Parse company and role from a combined string like 'Amazon SDE 1'
        
        Args:
            company_with_role: Combined string with company and role
            
        Returns:
            Tuple of (company, role, role_title)
        """
        try:
            # Clean and normalize the input
            normalized = company_with_role.strip()
            
            # Common company names to look for
            companies = ['Amazon', 'Google', 'Microsoft', 'Meta', 'Facebook', 'Apple', 'Netflix', 'Tesla', 'Uber', 'Airbnb', 'Spotify', 'Twitter', 'LinkedIn', 'Salesforce', 'Oracle', 'IBM', 'Intel', 'NVIDIA', 'Adobe', 'Shopify']
            
            company = None
            role_part = normalized
            
            # Find company name (case insensitive)
            for comp in companies:
                if comp.lower() in normalized.lower():
                    company = comp
                    # Remove company name from role part
                    role_part = re.sub(re.escape(comp), '', normalized, flags=re.IGNORECASE).strip()
                    break
            
            # If no known company found, take first word as company
            if not company:
                parts = normalized.split()
                if len(parts) > 1:
                    company = parts[0]
                    role_part = ' '.join(parts[1:])
                else:
                    company = "Unknown Company"
                    role_part = normalized
            
            # Generate role title from role part
            role_title = self._generate_role_title(role_part, company)
            
            return company, role_part, role_title
            
        except Exception as e:
            # Fallback parsing
            parts = company_with_role.split()
            company = parts[0] if parts else "Unknown"
            role = ' '.join(parts[1:]) if len(parts) > 1 else "Software Engineer"
            role_title = f"{role} at {company}"
            return company, role, role_title

    def _generate_role_title(self, role_part: str, company: str) -> str:
        """Generate a proper role title from role part"""
        role_mappings = {
            'sde': 'Software Development Engineer',
            'sde1': 'Software Development Engineer I',
            'sde2': 'Software Development Engineer II', 
            'sde3': 'Senior Software Development Engineer',
            'swe': 'Software Engineer',
            'swe1': 'Software Engineer',
            'swe2': 'Senior Software Engineer',
            'l3': 'Software Engineer L3',
            'l4': 'Software Engineer L4',
            'l5': 'Senior Software Engineer L5',
            'e3': 'Software Engineer E3',
            'e4': 'Software Engineer E4',
            'e5': 'Senior Software Engineer E5',
            'engineer': 'Software Engineer',
            'developer': 'Software Developer',
            'senior': 'Senior Software Engineer',
            'lead': 'Lead Software Engineer',
            'principal': 'Principal Software Engineer',
            'staff': 'Staff Software Engineer'
        }
        
        # Normalize role part
        role_lower = role_part.lower().strip()
        
        # Check for exact matches
        if role_lower in role_mappings:
            return role_mappings[role_lower]
        
        # Check for partial matches
        for key, value in role_mappings.items():
            if key in role_lower or role_lower in key:
                return value
        
        # If no match, return formatted version
        return role_part.title() if role_part else "Software Engineer"

    async def generate_screening_questions(self, company_with_role: str) -> Dict[str, Any]:
        """
        Generate comprehensive screening questions for any company and role dynamically
        
        Args:
            company_with_role: Combined string like "Amazon SDE 1" or "Google Software Engineer"
            
        Returns:
            Dictionary containing screening questions and metadata
        """
        try:
            company, role, role_title = self.parse_company_role(company_with_role)
            
            prompt = f"""
            Generate comprehensive screening questions for a {role_title} position at {company}.
            
            Based on the role "{role}" at "{company}", intelligently determine:
            1. Appropriate experience level expectations
            2. Relevant technical skills and technologies
            3. Company-specific culture and values
            4. Role-specific responsibilities and challenges
            
            Create exactly 12 screening questions covering these areas:
            1. Experience & Background (2 questions)
            2. Technical Skills & Programming Languages (3 questions) 
            3. Technologies & Tools (2 questions)
            4. Problem-Solving & Algorithms (2 questions)
            5. System Design & Architecture (1 question)
            6. Behavioral & Cultural Fit (2 questions)
            
            Make the questions:
            - Specific to {company}'s known culture and technology stack
            - Appropriate for the seniority level implied by "{role}"
            - Professional and comprehensive
            - Designed to assess real competency
            
            Format the response as a JSON object with this exact structure:
            {{
                "company": "{company}",
                "role": "{role}",
                "role_title": "{role_title}",
                "questions": [
                    {{
                        "id": 1,
                        "category": "Experience & Background",
                        "question": "question text here",
                        "type": "text"
                    }}
                ],
                "scoring_criteria": {{
                    "experience_weight": 20,
                    "technical_skills_weight": 30,
                    "problem_solving_weight": 25,
                    "behavioral_weight": 25
                }}
            }}
            
            Return ONLY the JSON object, no additional text or markdown formatting.
            """

            response = openai.ChatCompletion.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert technical recruiter. Generate comprehensive, role-specific screening questions in valid JSON format only. Do not include any markdown formatting or additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing non-JSON content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx]
            
            questions_data = json.loads(content)
            
            # Add metadata
            questions_data["generated_at"] = datetime.now().isoformat()
            questions_data["total_questions"] = len(questions_data["questions"])
            
            return questions_data
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            # Return fallback questions
            return self._generate_fallback_questions(company_with_role)

    def _generate_fallback_questions(self, company_with_role: str) -> Dict[str, Any]:
        """Generate fallback questions if AI generation fails"""
        company, role, role_title = self.parse_company_role(company_with_role)
        
        fallback_questions = [
            {
                "id": 1,
                "category": "Experience & Background",
                "question": f"How many years of software development experience do you have, and what types of projects have you worked on?",
                "type": "text"
            },
            {
                "id": 2,
                "category": "Experience & Background", 
                "question": f"What interests you about working at {company} in this {role_title} position?",
                "type": "text"
            },
            {
                "id": 3,
                "category": "Technical Skills",
                "question": "What programming languages are you most proficient in? Rate your proficiency from 1-10.",
                "type": "text"
            },
            {
                "id": 4,
                "category": "Technical Skills",
                "question": "Describe your experience with software development frameworks and libraries.",
                "type": "text"
            },
            {
                "id": 5,
                "category": "Technical Skills",
                "question": "What development tools and IDEs do you prefer and why?",
                "type": "text"
            },
            {
                "id": 6,
                "category": "Technologies & Tools",
                "question": "Describe your experience with cloud platforms and deployment.",
                "type": "text"
            },
            {
                "id": 7,
                "category": "Technologies & Tools",
                "question": "What database technologies have you worked with?",
                "type": "text"
            },
            {
                "id": 8,
                "category": "Problem Solving",
                "question": "Describe a challenging technical problem you solved recently. What was your approach?",
                "type": "text"
            },
            {
                "id": 9,
                "category": "Problem Solving",
                "question": "How do you approach debugging complex issues in production systems?",
                "type": "text"
            },
            {
                "id": 10,
                "category": "System Design",
                "question": "How would you design a scalable system to handle high traffic?",
                "type": "text"
            },
            {
                "id": 11,
                "category": "Behavioral",
                "question": "Describe a time when you had to work with a difficult team member. How did you handle it?",
                "type": "text"
            },
            {
                "id": 12,
                "category": "Behavioral",
                "question": f"Why do you want to work at {company} and what do you know about our culture?",
                "type": "text"
            }
        ]
        
        return {
            "company": company,
            "role": role,
            "role_title": role_title,
            "questions": fallback_questions,
            "scoring_criteria": {
                "experience_weight": 20,
                "technical_skills_weight": 30,
                "problem_solving_weight": 25,
                "behavioral_weight": 25
            },
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(fallback_questions)
        }

    async def assess_candidate_responses(self, company_with_role: str, questions: List[Dict[str, Any]], responses: Dict[int, str]) -> Dict[str, Any]:
        """
        Assess candidate responses and provide scoring and feedback
        
        Args:
            company_with_role: Combined company and role string
            questions: List of questions that were asked
            responses: Dictionary mapping question IDs to candidate responses
            
        Returns:
            Dictionary containing assessment results, scores, and feedback
        """
        try:
            company, role, role_title = self.parse_company_role(company_with_role)
            
            # Prepare responses for evaluation
            qa_pairs = []
            for question in questions:
                qid = question["id"]
                response = responses.get(qid, "No response provided")
                qa_pairs.append({
                    "question": question["question"],
                    "category": question["category"], 
                    "response": response
                })
            
            prompt = f"""
            Assess this candidate's responses for a {role_title} position at {company}.
            
            Role Context:
            - Company: {company}
            - Role: {role}
            - Position: {role_title}
            
            Question-Answer Pairs:
            {json.dumps(qa_pairs, indent=2)}
            
            Based on the role level and company, provide a comprehensive assessment considering:
            1. Technical competency appropriate for the role level
            2. Experience relevance to the position
            3. Cultural fit for {company}
            4. Communication skills and professionalism
            5. Problem-solving approach and thinking process
            
            Provide a thorough assessment in the following JSON format:
            {{
                "overall_score": 0-100,
                "category_scores": {{
                    "experience": 0-100,
                    "technical_skills": 0-100, 
                    "problem_solving": 0-100,
                    "behavioral": 0-100
                }},
                "strengths": ["list of candidate strengths"],
                "areas_for_improvement": ["list of areas needing improvement"],
                "detailed_feedback": {{
                    "experience": "detailed feedback on experience",
                    "technical_skills": "detailed feedback on technical skills",
                    "problem_solving": "detailed feedback on problem-solving",
                    "behavioral": "detailed feedback on behavioral aspects"
                }},
                "recommendation": "HIRE|MAYBE|NO_HIRE",
                "recommendation_reason": "detailed reason for the recommendation",
                "next_steps": ["list of recommended next steps"],
                "red_flags": ["any concerning responses or gaps"],
                "standout_responses": ["particularly impressive responses"]
            }}
            
            Be thorough, fair, and specific. Return ONLY the JSON object, no additional text.
            """

            response = openai.ChatCompletion.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": f"You are an expert technical recruiter assessing candidates for {company}. Provide detailed, fair, and actionable feedback in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing non-JSON content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx]
            
            assessment = json.loads(content)
            
            # Add metadata
            assessment["assessment_date"] = datetime.now().isoformat()
            assessment["company"] = company
            assessment["role"] = role
            assessment["role_title"] = role_title
            assessment["total_responses"] = len(responses)
            assessment["response_completion_rate"] = len([r for r in responses.values() if r and r.strip()]) / len(questions) * 100
            
            return assessment
            
        except Exception as e:
            print(f"Error assessing responses: {str(e)}")
            company, role, role_title = self.parse_company_role(company_with_role)
            # Return basic assessment
            return {
                "overall_score": 50,
                "category_scores": {
                    "experience": 50,
                    "technical_skills": 50,
                    "problem_solving": 50, 
                    "behavioral": 50
                },
                "strengths": ["Completed the screening"],
                "areas_for_improvement": ["Assessment could not be completed due to technical error"],
                "detailed_feedback": {
                    "experience": "Manual review required",
                    "technical_skills": "Manual review required",
                    "problem_solving": "Manual review required",
                    "behavioral": "Manual review required"
                },
                "recommendation": "MAYBE",
                "recommendation_reason": "Manual review required due to assessment error",
                "next_steps": ["Manual review by hiring manager"],
                "red_flags": [],
                "standout_responses": [],
                "error": str(e),
                "assessment_date": datetime.now().isoformat(),
                "company": company,
                "role": role,
                "role_title": role_title,
                "total_responses": len(responses),
                "response_completion_rate": 0
            }
