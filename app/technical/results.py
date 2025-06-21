from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Literal
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

# FastAPI app
app = FastAPI(title="Technical & Aptitude Answer Checker API", version="1.0.0")

# Pydantic models
class Question(BaseModel):
    question: str
    options: List[str]
    answer: str

class EvaluationRequest(BaseModel):
    questions: List[Question]
    question_type: Literal["aptitude", "technical", "os", "cn", "dbms"] = "aptitude"

class EvaluationResponse(BaseModel):
    overallScore: str
    feedback: Dict[str, Any]
    detailedResults: List[Dict[str, Any]]
    questionType: str

def generate_answer_groq(question: str, options: List[str], question_type: str) -> str:
    """Generate correct answer using Groq LLM based on question type"""
    formatted_options = "\n".join([f"- {opt}" for opt in options])
    
    # Different prompts for different question types
    if question_type == "aptitude":
        system_prompt = "You are an expert aptitude test evaluator specializing in quantitative aptitude, logical reasoning, and mathematical problem-solving."
    elif question_type == "technical":
        system_prompt = "You are an expert computer science technical interviewer with deep knowledge of programming concepts, algorithms, data structures, and software engineering principles."
    elif question_type == "os":
        system_prompt = "You are an expert in Operating Systems with comprehensive knowledge of process management, memory management, file systems, synchronization, deadlocks, and system calls."
    elif question_type == "cn":
        system_prompt = "You are an expert in Computer Networks with deep understanding of network protocols, OSI model, TCP/IP, routing, switching, network security, and distributed systems."
    elif question_type == "dbms":
        system_prompt = "You are an expert in Database Management Systems with extensive knowledge of relational databases, SQL, normalization, transactions, concurrency control, indexing, and query optimization."
    else:
        system_prompt = "You are an expert evaluator."

    prompt = f"""
{system_prompt}

Your task is to choose the correct answer to the multiple-choice question below. Select the correct answer **only** from the given options.

⚠️ Output must be **only** the matching option text exactly as listed, and **no explanation**, no dash, no prefix, and no extra characters.

Question: {question}

Options:
{formatted_options}

Only return the correct option, nothing else.
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

def categorize_technical_questions(results: List[Dict[str, Any]], question_type: str) -> Dict[str, Dict[str, int]]:
    """Categorize technical questions by topic"""
    if question_type == "aptitude":
        question_categories = {
            'speed_time_distance': ['train', 'speed', 'distance', 'time', 'km/hr', 'platform'],
            'percentage': ['percent', '%', 'percentage'],
            'arithmetic': ['runs', 'score', 'boundaries', 'sixes'],
            'profit_loss': ['profit', 'loss', 'cost price', 'selling price'],
            'ratios': ['ratio', 'proportion', 'share'],
            'geometry': ['area', 'perimeter', 'circle', 'triangle', 'rectangle']
        }
    elif question_type == "technical":
        question_categories = {
            'algorithms': ['algorithm', 'complexity', 'big o', 'sorting', 'searching', 'recursion'],
            'data_structures': ['array', 'linked list', 'stack', 'queue', 'tree', 'graph', 'hash'],
            'programming': ['variable', 'function', 'class', 'object', 'inheritance', 'polymorphism'],
            'software_engineering': ['design pattern', 'testing', 'debugging', 'version control']
        }
    elif question_type == "os":
        question_categories = {
            'process_management': ['process', 'thread', 'scheduling', 'context switch', 'fork'],
            'memory_management': ['memory', 'virtual memory', 'paging', 'segmentation', 'heap', 'stack'],
            'file_systems': ['file', 'directory', 'inode', 'disk', 'storage'],
            'synchronization': ['mutex', 'semaphore', 'deadlock', 'race condition', 'critical section'],
            'system_calls': ['system call', 'kernel', 'user mode', 'kernel mode']
        }
    elif question_type == "cn":
        question_categories = {
            'protocols': ['tcp', 'udp', 'http', 'ftp', 'smtp', 'dns', 'dhcp'],
            'osi_model': ['osi', 'layer', 'physical', 'data link', 'network', 'transport', 'session', 'presentation', 'application'],
            'routing': ['routing', 'router', 'switch', 'gateway', 'subnet'],
            'security': ['encryption', 'ssl', 'tls', 'firewall', 'vpn'],
            'network_concepts': ['bandwidth', 'latency', 'packet', 'frame', 'collision']
        }
    elif question_type == "dbms":
        question_categories = {
            'sql': ['select', 'insert', 'update', 'delete', 'join', 'group by', 'order by'],
            'normalization': ['normal form', 'normalization', '1nf', '2nf', '3nf', 'bcnf'],
            'transactions': ['transaction', 'acid', 'commit', 'rollback', 'isolation'],
            'indexing': ['index', 'primary key', 'foreign key', 'unique', 'clustered'],
            'query_optimization': ['query', 'optimization', 'execution plan', 'cost']
        }
    else:
        question_categories = {}
    
    category_performance = {}
    for result in results:
        question_lower = result['question'].lower()
        categorized = False
        
        for category, keywords in question_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                if category not in category_performance:
                    category_performance[category] = {'correct': 0, 'total': 0}
                category_performance[category]['total'] += 1
                if result['is_correct']:
                    category_performance[category]['correct'] += 1
                categorized = True
                break
        
        # If not categorized, put in 'other'
        if not categorized:
            if 'other' not in category_performance:
                category_performance['other'] = {'correct': 0, 'total': 0}
            category_performance['other']['total'] += 1
            if result['is_correct']:
                category_performance['other']['correct'] += 1
    
    return category_performance

def generate_detailed_feedback(results: List[Dict[str, Any]], score: float, question_type: str) -> Dict[str, Any]:
    """Generate detailed feedback based on results and question type"""
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    
    strengths = []
    improvements = []
    
    # Performance level assessment
    if score >= 80:
        strengths.append("Excellent overall performance")
        if question_type == "aptitude":
            strengths.append("Strong quantitative and logical reasoning skills")
        else:
            strengths.append(f"Strong grasp of {question_type.upper() if question_type in ['os', 'cn'] else question_type} fundamentals")
    elif score >= 60:
        strengths.append("Good understanding of basic concepts")
        strengths.append("Shows solid foundation with room for growth")
    else:
        improvements.append("Need to strengthen fundamental concepts")
        improvements.append("Requires more focused practice and study")
    
    # Category-specific analysis
    category_performance = categorize_technical_questions(results, question_type)
    
    # Add category-specific feedback
    for category, perf in category_performance.items():
        accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        if accuracy >= 0.8:
            strengths.append(f"Excellent performance in {category.replace('_', ' ')} concepts")
        elif accuracy < 0.5:
            improvements.append(f"Need improvement in {category.replace('_', ' ')} topics")
    
    # Default feedback if none specific
    if not strengths:
        strengths.append("Shows effort in attempting all questions")
    
    if not improvements:
        improvements.append("Continue practicing to maintain excellent performance")
    
    # Generate detailed feedback text
    performance_level = 'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Needs Improvement'
    
    detailed_feedback = f"""
Performance Summary:
- Question Type: {question_type.upper() if question_type in ['os', 'cn'] else question_type.title()}
- Answered {correct_count} out of {total_count} questions correctly ({score:.1f}% accuracy)
- Overall performance: {performance_level}

Category-wise Performance:
"""
    
    for category, perf in category_performance.items():
        accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        detailed_feedback += f"- {category.replace('_', ' ').title()}: {perf['correct']}/{perf['total']} ({accuracy*100:.1f}%)\n"
    
    # Question type specific recommendations
    if question_type == "aptitude":
        recommendation = "Focus on time management and practice more varied problem types" if score >= 70 else "Strengthen basic mathematical concepts and logical reasoning"
    elif question_type == "technical":
        recommendation = "Explore advanced algorithms and system design" if score >= 70 else "Practice coding problems and review fundamental programming concepts"
    elif question_type == "os":
        recommendation = "Study advanced OS topics like distributed systems" if score >= 70 else "Review core OS concepts: processes, memory management, and file systems"
    elif question_type == "cn":
        recommendation = "Learn about network security and advanced protocols" if score >= 70 else "Focus on OSI model, TCP/IP stack, and basic networking concepts"
    elif question_type == "dbms":
        recommendation = "Study NoSQL databases and advanced query optimization" if score >= 70 else "Practice SQL queries, normalization, and transaction concepts"
    else:
        recommendation = "Continue practicing regularly"
    
    detailed_feedback += f"\nRecommendations: {recommendation}"
    
    return {
        "strengths": strengths,
        "improvements": improvements,
        "detailedFeedback": detailed_feedback.strip()
    }

@app.post("/evaluate-answers", response_model=EvaluationResponse)
async def evaluate_answers(request: EvaluationRequest):
    """
    Evaluate answers using LLM-generated correct answers for different question types
    """
    try:
        results = []
        correct_answers = 0
        
        for question_data in request.questions:
            # Generate correct answer using LLM based on question type
            correct_answer = generate_answer_groq(
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
        
        # Generate feedback based on question type
        feedback = generate_detailed_feedback(results, overall_score, request.question_type)
        
        return EvaluationResponse(
            overallScore=f"{round(overall_score, 1)}%",
            feedback=feedback,
            detailedResults=results,
            questionType=request.question_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing evaluation: {str(e)}")

