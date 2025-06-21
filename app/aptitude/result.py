from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import openai
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure Groq API
openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"


# Pydantic models


def generate_answer_groq(question: str) -> str:
    """Generate correct answer using Groq LLM"""

    prompt = f"""
You are an expert aptitude test evaluator. Your task is to choose the correct answer to the multiple-choice question below. Select the correct answer **only** from the given options.

⚠️ Output must be **only** the matching option text exactly as listed, and **no explanation**, no dash, no prefix, and no extra characters.

Question: {question}


Only return the correct answer, nothing else.
"""

    try:
        response = openai.ChatCompletion.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = response['choices'][0]['message']['content'].strip()
        
        # Remove unwanted prefixes like '- ' just in case
        if answer.startswith("- "):
            answer = answer[2:].strip()
        
        return answer
    except Exception as e:
        print(f"❌ Error generating answer for question: {question[:50]}... -> {e}")
        return ""

def normalize_answer(answer: str) -> str:
    """Normalize answer by removing special characters and converting to lowercase"""
    # Remove special characters like $, extra spaces, etc.
    normalized = re.sub(r'[^\w\s.%]', '', answer.strip())
    return normalized.lower().replace(' ', '')

def is_answer_correct(user_answer: str, correct_answer: str) -> bool:
    """Check if user answer matches correct answer with fuzzy matching"""
    user_normalized = normalize_answer(user_answer)
    correct_normalized = normalize_answer(correct_answer)
    
    # Direct match
    if user_normalized == correct_normalized:
        return True
    
    # Check if user answer is contained in correct answer or vice versa
    if user_normalized in correct_normalized or correct_normalized in user_normalized:
        return True
    
    return False

def generate_detailed_feedback(results: List[Dict[str, Any]], score: float) -> Dict[str, Any]:
    """Generate detailed feedback based on results"""
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    
    strengths = []
    improvements = []
    
    # Analyze performance
    if score >= 80:
        strengths.append("Excellent overall performance")
        strengths.append("Strong problem-solving skills demonstrated")
    elif score >= 60:
        strengths.append("Good understanding of basic concepts")
        strengths.append("Shows potential for improvement")
    else:
        improvements.append("Need to focus on fundamental concepts")
        improvements.append("Practice more problem-solving exercises")
    
    # Analyze specific areas
    question_types = {
        'speed': ['train', 'speed', 'distance', 'time'],
        'percentage': ['percent', '%', 'percentage'],
        'arithmetic': ['runs', 'score', 'boundaries', 'sixes']
    }
    
    type_performance = {}
    for result in results:
        question_lower = result['question'].lower()
        for qtype, keywords in question_types.items():
            if any(keyword in question_lower for keyword in keywords):
                if qtype not in type_performance:
                    type_performance[qtype] = {'correct': 0, 'total': 0}
                type_performance[qtype]['total'] += 1
                if result['is_correct']:
                    type_performance[qtype]['correct'] += 1
    
    # Add type-specific feedback
    for qtype, perf in type_performance.items():
        accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        if accuracy >= 0.8:
            strengths.append(f"Strong performance in {qtype} problems")
        elif accuracy < 0.5:
            improvements.append(f"Need improvement in {qtype} problems")
    
    # Default feedback if none specific
    if not strengths:
        strengths.append("Shows effort in attempting all questions")
    
    if not improvements:
        improvements.append("Continue practicing to maintain performance")
    
    # Generate detailed feedback text
    detailed_feedback = f"""
Performance Summary:
- Answered {correct_count} out of {total_count} questions correctly ({score:.1f}% accuracy)
- Overall performance: {'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Needs Improvement'}

Key Areas:
"""
    
    for qtype, perf in type_performance.items():
        accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        detailed_feedback += f"- {qtype.title()} problems: {perf['correct']}/{perf['total']} ({accuracy*100:.1f}%)\n"
    
    detailed_feedback += f"\nRecommendations: Focus on {'advanced problem-solving techniques' if score >= 70 else 'fundamental concepts and regular practice'}."
    
    return {
        "strengths": strengths,
        "improvements": improvements,
        "detailedFeedback": detailed_feedback.strip()
    }



