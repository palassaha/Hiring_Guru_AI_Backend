from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
import os
import random
import time
import re
from itertools import product

class AptitudeQuestionScraper:
    def __init__(self, headless=True):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless=new")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        self.driver = None
        self.TOPIC_URLS = self._generate_random_topic_urls()

    def _generate_random_topic_urls(self):
        base_topics = [
            "problems-on-trains", "percentage", "average",
            "profit-and-loss", "time-and-distance"
        ]
        page_indices = random.sample(range(1, 6), 2)  # 2 random pages per topic
        urls = [
            f"https://www.indiabix.com/aptitude/{topic}/" for topic in base_topics
        ]
        random.shuffle(urls)
        return urls

    def start_driver(self):
        self.driver = webdriver.Chrome(options=self.options)

    def close_driver(self):
        if self.driver:
            self.driver.quit()

    def clean_question_text(self, text):
        if not text:
            return ""
        text = re.sub(r"^\d+\.\s*", "", text.strip())
        text = re.sub(r"^Question\s*\d*:?\s*", "", text, flags=re.IGNORECASE)
        return ' '.join(text.split())

    def parse_options(self, full_text):
        lines = full_text.split("\n")
        options = []
        for line in lines:
            if re.match(r"^[A-E]\)", line.strip()):
                options.append(re.sub(r"^[A-E]\)\s*", "", line.strip()))
        return options

    def extract_questions(self):
        elements = self.driver.find_elements(By.CSS_SELECTOR, ".bix-div-container")
        questions = []
        for elem in elements:
            full_text = elem.text.strip()
            lines = full_text.split("\n")
            question = ""
            options = []
            for line in lines:
                if "?" in line and not question:
                    question = self.clean_question_text(line)
                elif re.match(r"^[A-E]\)", line):
                    options.append(re.sub(r"^[A-E]\)\s*", "", line.strip()))
            if question and len(options) >= 2:
                questions.append({
                    "question": question,
                    "options": options,
                    "answer": "",
                    "explanation": ""
                })
        return questions

    def scrape_topic(self, url, max_per_topic=3):
        try:
            self.driver.get(url)
            time.sleep(2)
            questions = self.extract_questions()
            return questions[:max_per_topic]
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []

    def scrape_all_topics(self, max_per_topic=3):
        self.start_driver()
        all_questions = []
        try:
            for url in self.TOPIC_URLS:
                questions = self.scrape_topic(url, max_per_topic=max_per_topic)
                all_questions.extend(questions)
                time.sleep(random.uniform(1.5, 3))
        finally:
            self.close_driver()

        # Remove duplicates
        unique_questions = []
        seen = set()
        for q in all_questions:
            if q["question"] not in seen:
                seen.add(q["question"])
                unique_questions.append(q)
        return unique_questions

    def save_questions(self, questions, filename="app/aptitude/question_bank.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved {len(questions)} questions to {filename}")

# ----------- Main ------------
def main():
    print("🚀 Starting Aptitude Scraper...")
    scraper = AptitudeQuestionScraper(headless=True)
    questions = scraper.scrape_all_topics(max_per_topic=3)
    print(f"\n🔍 Total unique questions scraped: {len(questions)}\n")

    for i, q in enumerate(questions[:3]):
        print(f"Q{i+1}: {q['question']}")
        for j, opt in enumerate(q["options"], 1):
            print(f"   {j}. {opt}")
        print()

    scraper.save_questions(questions)

if __name__ == "__main__":
    main()
