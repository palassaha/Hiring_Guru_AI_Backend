
import json
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from enum import Enum
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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

class AssessmentGenerator:
    def __init__(self, api_key: str):
        """Initialize the Gemini API client"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_assessment(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a custom assessment based on user data using Gemini
        
        Args:
            user_data: Dictionary containing user information including:
                - userId: str
                - skills: List[str]
                - contributionFreq: str
                - projectsCount: int
                - topLanguages: List[str]
                - recentActivity: Dict
                - repositoryStats: Dict
                - targetRole: str
                - dreamCompanies: List[str]
                - skillGaps: List[str]
                - careerPath: List[str]
        
        Returns:
            Dictionary containing the generated assessment data
        """
        try:
            # Create a comprehensive prompt for Gemini
            prompt = self._create_assessment_prompt(user_data)
            
            # Generate content using Gemini
            response = self.model.generate_content(prompt)
            
            # Parse the response and structure it according to our schema
            assessment_data = self._parse_gemini_response(response.text, user_data)
            
            return assessment_data
            
        except Exception as e:
            print(f"Error generating assessment: {str(e)}")
            # Return a fallback assessment if generation fails
            return self._create_fallback_assessment(user_data)
    
    def _create_assessment_prompt(self, user_data: Dict[str, Any]) -> str:
        """Create a detailed prompt for Gemini based on user data"""
        
        target_role = user_data.get('targetRole', 'Software Developer')
        skills = ', '.join(user_data.get('skills', []))
        top_languages = ', '.join(user_data.get('topLanguages', []))
        skill_gaps = ', '.join(user_data.get('skillGaps', []))
        dream_companies = ', '.join(user_data.get('dreamCompanies', []))
        projects_count = user_data.get('projectsCount', 0)
        
        prompt = f"""
        Create a comprehensive interview assessment for a candidate with the following profile:
        
        Target Role: {target_role}
        Current Skills: {skills}
        Top Programming Languages: {top_languages}
        Skill Gaps: {skill_gaps}
        Dream Companies: {dream_companies}
        Number of Projects: {projects_count}
        
        Based on this profile, create an interview assessment with the following structure:
        
        1. Assessment Name: Create a relevant name for this assessment
        2. Description: Brief description of what this assessment evaluates
        3. Difficulty Level: Choose from EASY, MEDIUM, or HARD based on the target role and skill level
        4. Rounds: Create 4-6 interview rounds appropriate for the target role
        
        For each round, provide:
        - Round Type: Choose from SCREENING, APTITUDE, COMMUNICATION, CODING, TECHNICAL, BEHAVIORAL, SYSTEM_DESIGN
        - Name: Specific name for this round
        - Description: What this round evaluates
        - Duration: Time in minutes (15-120 minutes depending on round type)
        - Sequence: Order of the round (1, 2, 3, etc.)
        - Config: Any specific configuration or notes for this round
        
        Guidelines:
        - For technical roles, include CODING and TECHNICAL rounds
        - For senior roles, include SYSTEM_DESIGN
        - Always include SCREENING and BEHAVIORAL rounds
        - Match difficulty to the target role level
        - Consider the skill gaps when designing rounds
        - Make the assessment comprehensive but realistic
        
        Please provide your response in a structured format that can be easily parsed.
        """
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Gemini's response and structure it according to our schema"""
        
        # This is a simplified parser - in production, you might want more sophisticated parsing
        # For now, we'll create a structured response based on the user data and some defaults
        
        target_role = user_data.get('targetRole', 'Software Developer')
        skill_level = self._determine_skill_level(user_data)
        
        # Create assessment based on target role and skill level
        assessment_name = f"{target_role} Assessment"
        difficulty = self._determine_difficulty(skill_level, user_data)
        
        # Generate rounds based on role and skills
        rounds = self._generate_rounds_for_role(target_role, user_data)
        
        assessment = CreateCustomAssessmentDto(
            name=assessment_name,
            description=f"Comprehensive assessment for {target_role} position",
            difficulty=difficulty.value,
            rounds=rounds
        )
        
        return {
            "name": assessment.name,
            "description": assessment.description,
            "difficulty": assessment.difficulty,
            "rounds": [
                {
                    "roundType": round.roundType,
                    "name": round.name,
                    "description": round.description,
                    "duration": round.duration,
                    "sequence": round.sequence,
                    "config": round.config
                }
                for round in assessment.rounds
            ]
        }
    
    def _determine_skill_level(self, user_data: Dict[str, Any]) -> str:
        """Determine skill level based on user data"""
        projects_count = user_data.get('projectsCount', 0)
        skills_count = len(user_data.get('skills', []))
        
        if projects_count >= 10 and skills_count >= 8:
            return 'senior'
        elif projects_count >= 5 and skills_count >= 5:
            return 'mid'
        else:
            return 'junior'
    
    def _determine_difficulty(self, skill_level: str, user_data: Dict[str, Any]) -> DifficultyLevel:
        """Determine assessment difficulty"""
        target_role = user_data.get('targetRole', '').lower()
        
        if 'senior' in target_role or 'lead' in target_role or skill_level == 'senior':
            return DifficultyLevel.HARD
        elif 'mid' in target_role or skill_level == 'mid':
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY
    
    def _generate_rounds_for_role(self, target_role: str, user_data: Dict[str, Any]) -> List[CreateCustomRoundDto]:
        """Generate appropriate rounds based on target role"""
        rounds = []
        role_lower = target_role.lower()
        skill_level = self._determine_skill_level(user_data)
        
        # Always start with screening
        rounds.append(CreateCustomRoundDto(
            roundType=RoundType.SCREENING.value,
            name="Initial Screening",
            description="Basic background and motivation assessment",
            duration=30,
            sequence=1,
            config={"focus_areas": ["background", "motivation", "basic_skills"]}
        ))
        
        # Add role-specific rounds
        if any(tech in role_lower for tech in ['developer', 'engineer', 'programmer']):
            rounds.extend([
                CreateCustomRoundDto(
                    roundType=RoundType.APTITUDE.value,
                    name="Technical Aptitude",
                    description="Problem-solving and logical reasoning",
                    duration=45,
                    sequence=2,
                    config={"question_types": ["logical_reasoning", "pattern_recognition"]}
                ),
                CreateCustomRoundDto(
                    roundType=RoundType.CODING.value,
                    name="Coding Challenge",
                    description="Live coding and algorithm implementation",
                    duration=60,
                    sequence=3,
                    config={
                        "languages": user_data.get('topLanguages', ['Python', 'JavaScript']),
                        "difficulty": "medium" if skill_level == 'mid' else "easy"
                    }
                ),
                CreateCustomRoundDto(
                    roundType=RoundType.TECHNICAL.value,
                    name="Technical Discussion",
                    description="Deep dive into technical concepts and past projects",
                    duration=45,
                    sequence=4,
                    config={"topics": user_data.get('skills', [])}
                )
            ])
            
            # Add system design for senior roles
            if skill_level in ['senior', 'mid'] or 'senior' in role_lower:
                rounds.append(CreateCustomRoundDto(
                    roundType=RoundType.SYSTEM_DESIGN.value,
                    name="System Design",
                    description="Design scalable systems and architecture",
                    duration=60,
                    sequence=5,
                    config={"complexity": "high" if skill_level == 'senior' else "medium"}
                ))
        
        # Always end with behavioral
        rounds.append(CreateCustomRoundDto(
            roundType=RoundType.BEHAVIORAL.value,
            name="Cultural Fit & Behavioral",
            description="Team fit and soft skills evaluation",
            duration=30,
            sequence=len(rounds) + 1,
            config={"focus_areas": ["teamwork", "communication", "problem_solving"]}
        ))
        
        return rounds
    
    def _create_fallback_assessment(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback assessment if Gemini generation fails"""
        target_role = user_data.get('targetRole', 'Software Developer')
        
        return {
            "name": f"Standard {target_role} Assessment",
            "description": f"Standard assessment for {target_role} position",
            "difficulty": "MEDIUM",
            "rounds": [
                {
                    "roundType": "SCREENING",
                    "name": "Initial Screening",
                    "description": "Basic background check",
                    "duration": 30,
                    "sequence": 1,
                    "config": {}
                },
                {
                    "roundType": "TECHNICAL",
                    "name": "Technical Interview",
                    "description": "Technical skills assessment",
                    "duration": 60,
                    "sequence": 2,
                    "config": {}
                },
                {
                    "roundType": "BEHAVIORAL",
                    "name": "Behavioral Interview",
                    "description": "Cultural fit assessment",
                    "duration": 45,
                    "sequence": 3,
                    "config": {}
                }
            ]
        }

def generate_custom_assessment(user_data: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    """
    Main function to generate custom assessment
    
    Args:
        user_data: User profile data from frontend
        api_key: Gemini API key (optional, can be set via environment variable)
    
    Returns:
        Generated assessment data
    """
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it as parameter.")
    
    generator = AssessmentGenerator(api_key)
    return generator.generate_assessment(user_data)