from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import json
import google.generativeai as genai
from enum import Enum
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="Assessment Generator API", version="1.0.0")

# Pydantic models for request/response
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

class AssessmentGenerator:
    def __init__(self, api_key: str):
        """Initialize the Gemini API client"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_assessment(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a custom assessment based on user data using Gemini
        
        Args:
            user_data: Dictionary containing user information
        
        Returns:
            Dictionary containing the generated assessment data
        """
        try:
            # Create a comprehensive prompt for Gemini
            prompt = self._create_assessment_prompt(user_data)
            
            # Generate content using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000,
                )
            )
            
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
        skills = user_data.get('skills', [])
        top_languages = user_data.get('topLanguages', [])
        skill_gaps = user_data.get('skillGaps', [])
        dream_companies = user_data.get('dreamCompanies', [])
        projects_count = user_data.get('projectsCount', 0)
        contribution_freq = user_data.get('contributionFreq', 'medium')
        career_path = user_data.get('careerPath', [])
        
        prompt = f"""
        You are an expert interview designer. Create a comprehensive and personalized interview assessment for a candidate with the following detailed profile:

        CANDIDATE PROFILE:
        - Target Role: {target_role}
        - Current Skills: {', '.join(skills) if skills else 'Not specified'}
        - Top Programming Languages: {', '.join(top_languages) if top_languages else 'Not specified'}
        - Project Experience: {projects_count} projects
        - Contribution Frequency: {contribution_freq}
        - Skill Gaps to Address: {', '.join(skill_gaps) if skill_gaps else 'None identified'}
        - Dream Companies: {', '.join(dream_companies) if dream_companies else 'Not specified'}
        - Career Path: {', '.join(career_path) if career_path else 'Not specified'}

        REQUIREMENTS:
        1. Create EXACTLY 5-7 interview rounds that are highly customized to this specific candidate
        2. Each round should be tailored to assess specific aspects relevant to their profile
        3. Consider their skill gaps and target role requirements
        4. Include diverse round types to comprehensively evaluate the candidate

        AVAILABLE ROUND TYPES:
        - SCREENING: Initial background and basic qualification check
        - APTITUDE: Logical reasoning, problem-solving, analytical thinking
        - COMMUNICATION: Verbal and written communication skills
        - CODING: Programming challenges, algorithm implementation
        - TECHNICAL: Deep technical knowledge discussion
        - BEHAVIORAL: Soft skills, cultural fit, past experiences
        - SYSTEM_DESIGN: Architecture and system design capabilities

        For each round, provide the following information in this EXACT format:

        ROUND [NUMBER]:
        Type: [Choose from the available round types above]
        Name: [Specific descriptive name for this round]
        Description: [Detailed description of what this round evaluates and why it's important for this candidate]
        Duration: [Time in minutes: 15-120 minutes]
        Config: [Specific configuration details, focus areas, or special instructions for this round]

        ASSESSMENT DETAILS:
        Assessment Name: [Create a compelling name for this complete assessment]
        Assessment Description: [Brief description of what this assessment evaluates overall]
        Difficulty Level: [EASY/MEDIUM/HARD based on role seniority and complexity]

        IMPORTANT GUIDELINES:
        - Make each round highly specific to the candidate's profile
        - Address their skill gaps through targeted rounds
        - Consider the requirements of their target role and dream companies
        - Ensure the difficulty matches their experience level
        - Create a logical flow from basic to advanced rounds
        - Include at least one round that tests their strongest skills
        - Include at least one round that addresses their skill gaps

        Please generate a comprehensive assessment now:
        """
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Gemini's response and structure it according to our schema"""
        
        try:
            # Extract assessment details
            assessment_name = self._extract_field(response_text, "Assessment Name")
            assessment_description = self._extract_field(response_text, "Assessment Description")
            difficulty = self._extract_field(response_text, "Difficulty Level")
            
            # Parse rounds from the response
            rounds = self._parse_rounds_from_response(response_text)
            
            # Ensure we have at least 5 rounds
            if len(rounds) < 5:
                rounds.extend(self._generate_additional_rounds(user_data, len(rounds)))
            
            # Validate and set defaults if extraction failed
            if not assessment_name:
                assessment_name = f"Custom {user_data.get('targetRole', 'Professional')} Assessment"
            
            if not assessment_description:
                assessment_description = f"Comprehensive evaluation for {user_data.get('targetRole', 'the target position')}"
            
            if not difficulty or difficulty.upper() not in ['EASY', 'MEDIUM', 'HARD']:
                difficulty = self._determine_difficulty_from_profile(user_data)
            
            return {
                "name": assessment_name,
                "description": assessment_description,
                "difficulty": difficulty.upper(),
                "rounds": [
                    {
                        "roundType": round.roundType,
                        "name": round.name,
                        "description": round.description,
                        "duration": round.duration,
                        "sequence": round.sequence,
                        "config": round.config
                    }
                    for round in rounds
                ]
            }
            
        except Exception as e:
            print(f"Error parsing Gemini response: {str(e)}")
            return self._create_fallback_assessment(user_data)
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a specific field from the Gemini response"""
        pattern = rf"{field_name}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _parse_rounds_from_response(self, response_text: str) -> List[CreateCustomRoundDto]:
        """Parse individual rounds from Gemini's response"""
        rounds = []
        
        # Find all round blocks
        round_pattern = r"ROUND\s+(\d+):(.*?)(?=ROUND\s+\d+:|ASSESSMENT DETAILS:|$)"
        round_matches = re.findall(round_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        for i, (round_num, round_content) in enumerate(round_matches):
            try:
                round_type = self._extract_round_field(round_content, "Type")
                name = self._extract_round_field(round_content, "Name")
                description = self._extract_round_field(round_content, "Description")
                duration_str = self._extract_round_field(round_content, "Duration")
                config_str = self._extract_round_field(round_content, "Config")
                
                # Parse duration
                duration = self._parse_duration(duration_str)
                
                # Parse config
                config = self._parse_config(config_str)
                
                # Validate round type
                if round_type.upper() not in [rt.value for rt in RoundType]:
                    round_type = self._map_to_valid_round_type(round_type)
                
                rounds.append(CreateCustomRoundDto(
                    roundType=round_type.upper(),
                    name=name or f"Round {i + 1}",
                    description=description or "Assessment round",
                    duration=duration,
                    sequence=i + 1,
                    config=config
                ))
                
            except Exception as e:
                print(f"Error parsing round {round_num}: {str(e)}")
                continue
        
        return rounds
    
    def _extract_round_field(self, round_content: str, field_name: str) -> str:
        """Extract a specific field from a round's content"""
        pattern = rf"{field_name}:\s*(.+?)(?:\n\w+:|$)"
        match = re.search(pattern, round_content, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration from string, extracting number of minutes"""
        if not duration_str:
            return 45  # Default duration
        
        # Extract number from string
        duration_match = re.search(r'(\d+)', duration_str)
        if duration_match:
            duration = int(duration_match.group(1))
            # Ensure reasonable bounds
            return max(15, min(120, duration))
        
        return 45  # Default fallback
    
    def _parse_config(self, config_str: str) -> Dict[str, Any]:
        """Parse configuration from string"""
        if not config_str:
            return {}
        
        try:
            # Try to extract key information from the config string
            config = {}
            
            # Look for common patterns
            if "focus" in config_str.lower():
                config["focus_areas"] = [area.strip() for area in config_str.split(",")]
            elif "language" in config_str.lower():
                config["languages"] = [lang.strip() for lang in config_str.split(",")]
            elif "topic" in config_str.lower():
                config["topics"] = [topic.strip() for topic in config_str.split(",")]
            else:
                config["details"] = config_str.strip()
            
            return config
            
        except Exception:
            return {"details": config_str.strip()}
    
    def _map_to_valid_round_type(self, round_type: str) -> str:
        """Map potentially invalid round type to a valid one"""
        round_type_lower = round_type.lower()
        
        if any(word in round_type_lower for word in ["screen", "initial", "background"]):
            return RoundType.SCREENING.value
        elif any(word in round_type_lower for word in ["code", "coding", "programming"]):
            return RoundType.CODING.value
        elif any(word in round_type_lower for word in ["technical", "tech"]):
            return RoundType.TECHNICAL.value
        elif any(word in round_type_lower for word in ["behavioral", "behavior", "soft"]):
            return RoundType.BEHAVIORAL.value
        elif any(word in round_type_lower for word in ["system", "design", "architecture"]):
            return RoundType.SYSTEM_DESIGN.value
        elif any(word in round_type_lower for word in ["aptitude", "logical", "reasoning"]):
            return RoundType.APTITUDE.value
        elif any(word in round_type_lower for word in ["communication", "presentation"]):
            return RoundType.COMMUNICATION.value
        else:
            return RoundType.TECHNICAL.value  # Default fallback
    
    def _generate_additional_rounds(self, user_data: Dict[str, Any], current_count: int) -> List[CreateCustomRoundDto]:
        """Generate additional rounds if we don't have enough"""
        additional_rounds = []
        target_role = user_data.get('targetRole', '').lower()
        skills = user_data.get('skills', [])
        skill_gaps = user_data.get('skillGaps', [])
        
        # Add essential rounds if missing
        if current_count < 5:
            rounds_to_add = [
                CreateCustomRoundDto(
                    roundType=RoundType.SCREENING.value,
                    name="Initial Screening",
                    description="Background verification and basic qualification check",
                    duration=30,
                    sequence=current_count + 1,
                    config={"focus_areas": ["background", "motivation", "availability"]}
                ),
                CreateCustomRoundDto(
                    roundType=RoundType.TECHNICAL.value,
                    name="Technical Expertise",
                    description="Deep dive into technical skills and knowledge",
                    duration=60,
                    sequence=current_count + 2,
                    config={"topics": skills[:5] if skills else ["general_technical"]}
                ),
                CreateCustomRoundDto(
                    roundType=RoundType.CODING.value,
                    name="Coding Challenge",
                    description="Practical programming assessment",
                    duration=75,
                    sequence=current_count + 3,
                    config={"languages": user_data.get('topLanguages', ['Python'])}
                ),
                CreateCustomRoundDto(
                    roundType=RoundType.BEHAVIORAL.value,
                    name="Behavioral Assessment",
                    description="Soft skills and cultural fit evaluation",
                    duration=45,
                    sequence=current_count + 4,
                    config={"focus_areas": ["teamwork", "leadership", "problem_solving"]}
                ),
                CreateCustomRoundDto(
                    roundType=RoundType.SYSTEM_DESIGN.value,
                    name="System Design",
                    description="Architectural thinking and scalability assessment",
                    duration=60,
                    sequence=current_count + 5,
                    config={"complexity": "medium", "focus": "scalability"}
                )
            ]
            
            # Add rounds until we reach at least 5
            needed = 5 - current_count
            additional_rounds.extend(rounds_to_add[:needed])
        
        return additional_rounds
    
    def _determine_difficulty_from_profile(self, user_data: Dict[str, Any]) -> str:
        """Determine difficulty based on user profile"""
        target_role = user_data.get('targetRole', '').lower()
        projects_count = user_data.get('projectsCount', 0)
        skills_count = len(user_data.get('skills', []))
        
        # Senior level indicators
        if any(word in target_role for word in ['senior', 'lead', 'principal', 'architect']):
            return 'HARD'
        elif projects_count >= 10 and skills_count >= 8:
            return 'HARD'
        # Mid level indicators
        elif any(word in target_role for word in ['mid', 'intermediate']) or projects_count >= 5:
            return 'MEDIUM'
        # Junior level
        else:
            return 'EASY'
    
    def _create_fallback_assessment(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive fallback assessment if Gemini generation fails"""
        target_role = user_data.get('targetRole', 'Software Developer')
        skills = user_data.get('skills', [])
        skill_gaps = user_data.get('skillGaps', [])
        top_languages = user_data.get('topLanguages', ['Python'])
        
        fallback_rounds = [
            CreateCustomRoundDto(
                roundType=RoundType.SCREENING.value,
                name="Initial Screening Call",
                description="Background verification and role alignment check",
                duration=30,
                sequence=1,
                config={"focus_areas": ["background", "motivation", "role_fit"]}
            ),
            CreateCustomRoundDto(
                roundType=RoundType.APTITUDE.value,
                name="Problem Solving Assessment",
                description="Logical reasoning and analytical thinking evaluation",
                duration=45,
                sequence=2,
                config={"question_types": ["logical_reasoning", "pattern_recognition", "analytical"]}
            ),
            CreateCustomRoundDto(
                roundType=RoundType.CODING.value,
                name="Live Coding Session",
                description="Practical programming skills assessment",
                duration=75,
                sequence=3,
                config={
                    "languages": top_languages,
                    "difficulty": "medium",
                    "focus_areas": ["algorithm_implementation", "code_quality"]
                }
            ),
            CreateCustomRoundDto(
                roundType=RoundType.TECHNICAL.value,
                name="Technical Deep Dive",
                description="In-depth discussion of technical expertise and past projects",
                duration=60,
                sequence=4,
                config={
                    "topics": skills[:5] if skills else ["general_programming"],
                    "skill_gaps": skill_gaps[:3] if skill_gaps else []
                }
            ),
            CreateCustomRoundDto(
                roundType=RoundType.SYSTEM_DESIGN.value,
                name="System Architecture Discussion",
                description="Evaluate system design and scalability thinking",
                duration=60,
                sequence=5,
                config={"complexity": "medium", "focus": ["scalability", "architecture"]}
            ),
            CreateCustomRoundDto(
                roundType=RoundType.BEHAVIORAL.value,
                name="Cultural Fit & Leadership",
                description="Soft skills, team dynamics, and cultural alignment assessment",
                duration=45,
                sequence=6,
                config={
                    "focus_areas": ["teamwork", "communication", "leadership", "adaptability"],
                    "scenario_based": True
                }
            )
        ]
        
        return {
            "name": f"Comprehensive {target_role} Assessment",
            "description": f"Multi-round evaluation designed for {target_role} candidates with focus on technical excellence and cultural fit",
            "difficulty": self._determine_difficulty_from_profile(user_data),
            "rounds": [
                {
                    "roundType": round.roundType,
                    "name": round.name,
                    "description": round.description,
                    "duration": round.duration,
                    "sequence": round.sequence,
                    "config": round.config
                }
                for round in fallback_rounds
            ]
        }

# Dependency to get API key
def get_api_key():
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY environment variable is not set"
        )
    return GEMINI_API_KEY

# Initialize generator
generator = None

def get_generator(api_key: str = Depends(get_api_key)):
    global generator
    if not generator:
        generator = AssessmentGenerator(api_key)
    return generator

