
import json
import re
import requests
from bs4 import BeautifulSoup
import time
import random

# (All other functions remain unchanged: get_all_questions_by_difficulty, get_leetcode_question, parse_question_content, etc.)
# Only scrape_random_questions and main logic are rewritten.

def scrape_random_questions(num_questions=2):
    """
    Scrape exactly `num_questions` random LeetCode problems from all difficulties.

    Returns:
        list: List of scraped questions
    """
    print(f"Fetching all available LeetCode questions...")
    all_questions = get_all_questions_by_difficulty()
    
    if not all_questions:
        print("❌ Failed to fetch questions list.")
        return {"error": "Could not fetch questions list"}
    
    # Combine all questions into one flat list
    combined_questions = all_questions['Easy'] + all_questions['Medium'] + all_questions['Hard']
    print(f"Total free questions found: {len(combined_questions)}")
    
    # Select random questions
    selected = random.sample(combined_questions, min(num_questions, len(combined_questions)))

    print(f"\nSelected {len(selected)} question(s): {[q['title'] for q in selected]}")
    
    results = []
    
    for idx, question in enumerate(selected, 1):
        print(f"[{idx}/{len(selected)}] Scraping: {question['title']}")
        detailed = get_leetcode_question(question['titleSlug'])
        if 'error' not in detailed:
            results.append(detailed)
            print(f"✓ Successfully scraped: {question['title']}")
        else:
            print(f"✗ Failed to scrape: {question['title']} - {detailed['error']}")
        time.sleep(1)
    
    return results

def save_results_to_file(results, filename="./app/dsa_coding/random_leetcode_questions.json"):
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {filename}")

def print_summary(results):
    print("\n" + "="*60)
    print("SCRAPING SUMMARY")
    print("="*60)
    
    print(f"Total: {len(results)} question(s) scraped")
    for q in results:
        print(f"   - {q.get('title', 'Unknown')} (#{q.get('frontend_id', 'N/A')})")

if __name__ == "__main__":
    print("LeetCode Random Question Scraper")
    print("=" * 40)
    print("Fetching exactly 2 random questions (any difficulty)...")
    
    questions = scrape_random_questions(num_questions=2)
    
    if isinstance(questions, list):
        save_results_to_file(questions)
        print_summary(questions)

        # Save flattened version
        with open("./app/dsa_coding/all_random_questions.json", "w", encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

        print(f"\n📝 Flattened version saved to all_random_questions.json")
    else:
        print("❌ Failed to scrape questions:", questions['error'])
