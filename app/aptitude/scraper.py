from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
import json
import os
import random
import time
import re

class AptitudeQuestionScraper:
    def __init__(self, headless=False):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        self.driver = None
        self.wait = None

        self.TOPIC_URLS = [
            "https://www.indiabix.com/aptitude/problems-on-trains/",
            "https://www.indiabix.com/aptitude/percentage/",
            "https://www.indiabix.com/aptitude/average/",
            "https://www.indiabix.com/aptitude/profit-and-loss/",
            "https://www.indiabix.com/aptitude/time-and-distance/"
        ]

        self.QUESTION_SELECTORS = [
            "div.bix-div-container",
            "tr.bix-div-container",
            ".bix-div-container",
            "div.bix-td-qtxt",
            "div.bix-td-qno",
            ".question-container",
            "tr[class*='question']",
            "div[class*='question']"
        ]

    def start_driver(self):
        self.driver = webdriver.Chrome(options=self.options)
        self.wait = WebDriverWait(self.driver, 15)

    def close_driver(self):
        if self.driver:
            self.driver.quit()

    def wait_for_page_load(self):
        try:
            self.wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")
            time.sleep(3)
        except TimeoutException:
            print("Page load timeout, continuing anyway...")

    def handle_popups_and_ads(self):
        try:
            close_selectors = [
                "button[class*='close']",
                ".close-button",
                ".modal-close",
                "[aria-label='Close']",
                ".popup-close",
                ".ad-close"
            ]
            for selector in close_selectors:
                try:
                    close_btn = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if close_btn.is_displayed():
                        close_btn.click()
                        time.sleep(1)
                except:
                    continue
        except:
            pass

    def clean_question_text(self, text):
        if not text:
            return ""
        text = re.sub(r'^\d+\.\s*', '', text.strip())
        text = ' '.join(text.split())
        text = re.sub(r'^Question\s*\d*:?:?\s*', '', text, flags=re.IGNORECASE)
        return text

    def improved_option_parser(self, text):
        if not text:
            return []

        options = []
        pattern1 = r'(?=[A-E]\)|[1-5]\))'
        parts = re.split(pattern1, text)

        for part in parts:
            part = part.strip()
            if not part:
                continue
            cleaned = re.sub(r'^[A-E]\)|^[1-5]\)', '', part).strip()
            if cleaned and len(cleaned) > 1:
                options.append(cleaned)

        if len(options) < 2:
            options = []
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if '?' in line:
                    continue
                cleaned = re.sub(r'^[A-E]\)|^[1-5]\)|^[A-E]\.|^[1-5]\.', '', line).strip()
                if cleaned and 1 < len(cleaned) < 200 and not cleaned.isdigit():
                    options.append(cleaned)

        if len(options) < 2:
            options = []
            percentage_pattern = r'(\d+\.?\d*%?|\d+/\d+)'
            matches = re.findall(percentage_pattern, text)
            if len(matches) >= 3:
                options = matches[:5]

        if len(options) < 2:
            options = []
            parts = re.split(r'\s{3,}|\t+', text)
            for part in parts:
                part = part.strip()
                if part and len(part) > 1 and '?' not in part:
                    cleaned = re.sub(r'^[A-E]\)|^[1-5]\)|^[A-E]\.|^[1-5]\.', '', part).strip()
                    if cleaned:
                        options.append(cleaned)

        final_options = []
        for opt in options:
            opt = ' '.join(opt.split())
            opt = re.sub(r'(\d+)\s+(\d+)\s*%', r'\1.\2%', opt)
            opt = re.sub(r'(\d+)\s+(\d+)\s*/\s*(\d+)', r'\1\2/\3', opt)
            if opt and opt not in final_options:
                final_options.append(opt)

        return final_options[:5]

    def extract_question_data(self, question_element):
        try:
            text = question_element.text.strip()
            lines = text.split('\n')
            question_candidates = [line for line in lines if '?' in line and len(line) > 10]
            question_text = self.clean_question_text(question_candidates[0]) if question_candidates else ""
            options = self.improved_option_parser(text)
            return {"question": question_text, "options": options, "answer": "", "explanation": ""} if question_text and len(options) >= 2 else None
        except Exception as e:
            print(f"Error extracting question data: {e}")
            return None

    def scrape_topic(self, url, max_per_topic=3):
        print(f"\nScraping {url}")
        try:
            self.driver.get(url)
            self.wait_for_page_load()
            self.handle_popups_and_ads()
            print(f"Page title: {self.driver.title}")
            questions = []
            for selector in self.QUESTION_SELECTORS:
                try:
                    question_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    print(f"Selector '{selector}': found {len(question_elements)} elements")
                    random.shuffle(question_elements)
                    for i, elem in enumerate(question_elements[:max_per_topic]):
                        print(f"\nProcessing element {i+1}")
                        question_data = self.extract_question_data(elem)
                        if question_data:
                            questions.append(question_data)
                            print(f"\u2713 Extracted: {question_data['question'][:60]}...")
                        else:
                            print(f"\u2717 Skipped element {i+1}")
                    if questions:
                        print(f"\u2705 Found {len(questions)} questions with selector '{selector}'")
                        break
                except Exception as e:
                    print(f"Selector error '{selector}': {e}")
                    continue
            return questions
        except Exception as e:
            print(f"Scrape error on {url}: {e}")
            return []

    def scrape_all_topics(self, max_per_topic=3):
        all_questions = []
        try:
            self.start_driver()
            for url in self.TOPIC_URLS:
                questions = self.scrape_topic(url, max_per_topic=max_per_topic)
                if questions:
                    questions = questions[:max_per_topic]
                    all_questions.extend(questions)
                    print(f"Added {len(questions)} questions from this topic")
                time.sleep(random.uniform(3, 6))
        except Exception as e:
            print(f"Error during scraping: {e}")
        finally:
            self.close_driver()
        return all_questions

    def save_questions(self, questions, filename="app/aptitude/question_bank.json"):
        if not questions:
            print("No questions to save.")
            return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(questions)} questions to {filename}")

    # Fixed: Made this an instance method instead of static
    def run_scraping(self):
        """Run the complete scraping process"""
        print("Starting Aptitude Question Scraper...")
        all_questions = self.scrape_all_topics(max_per_topic=2)
        print(f"\n=== RESULTS ===")
        print(f"Total questions found: {len(all_questions)}")
        if all_questions:
            self.save_questions(all_questions)
            print(f"\n=== SAMPLE QUESTIONS ===")
            for i, q in enumerate(all_questions[:3]):
                print(f"\nQuestion {i+1}:")
                print(f"Q: {q['question']}")
                print(f"Options ({len(q['options'])}):")
                for j, opt in enumerate(q['options'], 1):
                    print(f"  {j}. {opt}")
                if q['answer']:
                    print(f"Answer: {q['answer']}")
            return all_questions
        else:
            print("\nNo questions extracted. Possible issues:")
            print("1. Website structure has changed")
            print("2. Anti-bot protection is blocking access")
            print("3. Network connectivity issues")
            print("4. Selectors need updating")
            print("\nTroubleshooting suggestions:")
            print("1. Run with headless=False to see browser behavior")
            print("2. Check the website manually")
            print("3. Add more delays")
            print("4. Use a different approach (API, manual creation, etc.)")
            return []

# Fixed: Added the missing main function
def main():
    """Main function to run the scraper"""
    scraper = AptitudeQuestionScraper(headless=False)
    scraper.run_scraping()

if __name__ == "__main__":
    main()