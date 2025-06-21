import json
import re
import requests
from bs4 import BeautifulSoup
import time
import random

def get_all_questions_by_difficulty():
    """
    Get all questions categorized by difficulty using GraphQL API
    
    Returns:
        dict: Questions organized by difficulty
    """
    
    query = """
    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
        problemsetQuestionList: questionList(
            categorySlug: $categorySlug
            limit: $limit
            skip: $skip
            filters: $filters
        ) {
            total: totalNum
            questions: data {
                acRate
                difficulty
                freqBar
                frontendQuestionId: questionFrontendId
                isFavor
                paidOnly: isPaidOnly
                status
                title
                titleSlug
                topicTags {
                    name
                    id
                    slug
                }
            }
        }
    }
    """
    
    url = "https://leetcode.com/graphql"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    payload = {
        "query": query,
        "variables": {
            "categorySlug": "",
            "skip": 0,
            "limit": 3000, 
            "filters": {}
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('data') and data['data'].get('problemsetQuestionList'):
                questions = data['data']['problemsetQuestionList']['questions']
                
                free_questions = {
                    'Easy': [],
                    'Medium': [],
                    'Hard': []
                }
                
                for question in questions:
                    if not question.get('paidOnly', True):
                        difficulty = question['difficulty']
                        if difficulty in free_questions:
                            free_questions[difficulty].append({
                                'title': question['title'],
                                'titleSlug': question['titleSlug'],
                                'difficulty': difficulty,
                                'frontendQuestionId': question['frontendQuestionId']
                            })
                
                return free_questions
            else:
                print("Error: Could not fetch questions list")
                return None
                
    except Exception as e:
        print(f"Error fetching questions list: {str(e)}")
        return None

def get_leetcode_question(title_slug):
    """
    Get LeetCode question data using GraphQL API
    
    Args:
        title_slug (str): Question slug like 'two-sum'
    
    Returns:
        dict: Structured question data
    """

    query = """
    query getQuestionDetail($titleSlug: String!) {
        question(titleSlug: $titleSlug) {
            questionId
            questionFrontendId
            title
            titleSlug
            content
            difficulty
            topicTags {
                name
            }
        }
    }
    """

    url = "https://leetcode.com/graphql"
    
    payload = {
        "query": query,
        "variables": {"titleSlug": title_slug}
    }
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('data') and data['data'].get('question'):
                question_data = data['data']['question']
                
                parsed_data = parse_question_content(question_data['content'])
                
                result = {
                    "question_id": question_data['questionId'],
                    "frontend_id": question_data['questionFrontendId'],
                    "title": question_data['title'],
                    "title_slug": question_data['titleSlug'],
                    "difficulty": question_data['difficulty'],
                    "topics": [tag['name'] for tag in question_data.get('topicTags', [])],
                    "problem_statement": parsed_data['problem_statement'],
                    "examples": parsed_data['examples'],
                    "constraints": parsed_data['constraints'],
                    "input_format": parsed_data['input_format'],
                    "output_format": parsed_data['output_format']
                }
                
                return result
            else:
                return {"error": "Question not found"}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def parse_question_content(html_content):
    """
    Parse HTML content to extract structured information
    """
    if not html_content:
        return {
            "problem_statement": "",
            "examples": [],
            "constraints": [],
            "input_format": "",
            "output_format": ""
        }
    
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    result = {
        "problem_statement": extract_problem_statement(text),
        "examples": extract_examples(text),
        "constraints": extract_constraints(text),
        "input_format": extract_input_format(text),
        "output_format": extract_output_format(text)
    }
    
    return result

def extract_problem_statement(text):
    """Extract the main problem statement"""
    match = re.search(r'^(.*?)(?=Example\s*\d*:|Constraints?:|$)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text[:500] + "..." if len(text) > 500 else text

def extract_examples(text):
    """Extract examples from the text"""
    examples = []
    
    pattern = r'Example\s*(\d*):(.*?)(?=Example\s*\d*:|Constraints?:|Note:|$)'
    matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
    
    for i, match in enumerate(matches, 1):
        example_text = match.group(2).strip()
        
        input_match = re.search(r'Input:\s*(.*?)(?=Output:|Explanation:|$)', example_text, re.DOTALL)
        input_text = input_match.group(1).strip() if input_match else ""
        
        output_match = re.search(r'Output:\s*(.*?)(?=Explanation:|$)', example_text, re.DOTALL)
        output_text = output_match.group(1).strip() if output_match else ""
        
        explanation_match = re.search(r'Explanation:\s*(.*?)$', example_text, re.DOTALL)
        explanation_text = explanation_match.group(1).strip() if explanation_match else ""
        
        example = {
            "example_number": i,
            "input": input_text,
            "output": output_text,
            "explanation": explanation_text
        }
        examples.append(example)
    
    return examples

def extract_constraints(text):
    """Extract constraints from the text"""
    constraints = []
    
    match = re.search(r'Constraints?:(.*?)(?=Example|Note:|Follow[- ]?up:|$)', text, re.DOTALL | re.IGNORECASE)
    
    if match:
        constraints_text = match.group(1).strip()
        lines = constraints_text.split('\n')
        for line in lines:
            line = line.strip()
            line = re.sub(r'^[-•*]\s*', '', line)
            if line and not line.lower().startswith('example'):
                constraints.append(line)
    
    return constraints

def extract_input_format(text):
    """Extract input format description"""
    patterns = [
        r'Input:\s*(.*?)(?=Output:|Example|Constraints|$)',
        r'Input\s*Format:\s*(.*?)(?=Output|Example|Constraints|$)',
        r'The\s*input\s*(?:is|consists of):\s*(.*?)(?=Output|Example|Constraints|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""

def extract_output_format(text):
    """Extract output format description"""
    patterns = [
        r'Output:\s*(.*?)(?=Example|Constraints|Note|$)',
        r'Output\s*Format:\s*(.*?)(?=Example|Constraints|Note|$)',
        r'Return\s*(.*?)(?=Example|Constraints|Note|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""

def scrape_random_questions():
    """
    Main function to scrape random questions:
    - 4 Easy questions
    - 4 Medium questions  
    - 2 Hard questions
    
    Returns:
        dict: All scraped questions organized by difficulty
    """
    print("Fetching all available questions...")
    
    all_questions = get_all_questions_by_difficulty()
    
    if not all_questions:
        print("Failed to fetch questions list")
        return {"error": "Could not fetch questions list"}
    
    print(f"Found {len(all_questions['Easy'])} Easy, {len(all_questions['Medium'])} Medium, {len(all_questions['Hard'])} Hard questions")
    
    selected_questions = {
        'Easy': random.sample(all_questions['Easy'], min(2, len(all_questions['Easy']))),
        'Medium': random.sample(all_questions['Medium'], min(2, len(all_questions['Medium']))),
        'Hard': random.sample(all_questions['Hard'], min(1, len(all_questions['Hard'])))
    }
    
    print("\nSelected random questions:")
    for difficulty, questions in selected_questions.items():
        print(f"{difficulty}: {[q['title'] for q in questions]}")
    results = {
        'Easy': [],
        'Medium': [],
        'Hard': []
    }
    
    total_questions = sum(len(questions) for questions in selected_questions.values())
    current_question = 0
    
    for difficulty, questions in selected_questions.items():
        print(f"\nScraping {difficulty} questions...")
        
        for question in questions:
            current_question += 1
            print(f"[{current_question}/{total_questions}] Scraping: {question['title']}")
            
            detailed_question = get_leetcode_question(question['titleSlug'])
            
            if 'error' not in detailed_question:
                results[difficulty].append(detailed_question)
                print(f"✓ Successfully scraped: {question['title']}")
            else:
                print(f"✗ Failed to scrape: {question['title']} - {detailed_question['error']}")
            time.sleep(1)
    
    return results

def save_results_to_file(results, filename="./app/dsa_coding/random_leetcode_questions.json"):
    """Save results to JSON file"""
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {filename}")

def print_summary(results):
    """Print a summary of scraped questions"""
    print("\n" + "="*60)
    print("SCRAPING SUMMARY")
    print("="*60)
    
    total_scraped = 0
    for difficulty, questions in results.items():
        valid_questions = [q for q in questions if 'error' not in q]
        total_scraped += len(valid_questions)
        print(f"{difficulty:8}: {len(valid_questions)} questions scraped")
        
        for question in valid_questions:
            print(f"   - {question.get('title', 'Unknown')} (#{question.get('frontend_id', 'N/A')})")
    
    print(f"\nTotal: {total_scraped} questions successfully scraped")


if __name__ == "__main__":
    print("LeetCode Random Questions Scraper")
    print("=" * 40)
    print("Fetching 4 Easy + 4 Medium + 2 Hard questions...")
    results = scrape_random_questions()
    
    if 'error' not in results:
        save_results_to_file(results)
        print_summary(results)
        
        all_questions = []
        for difficulty_questions in results.values():
            all_questions.extend(difficulty_questions)
        
        with open("./app/dsa_coding/all_random_questions.json", "w", encoding='utf-8') as f:
            json.dump(all_questions, f, indent=2, ensure_ascii=False)
        
        print(f"\nFlattened version saved to all_random_questions.json")
        print(f"Total questions in output: {len(all_questions)}")
        
    else:
        print("Failed to scrape questions:", results['error'])