import os
import json
import time
import random
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

class LeetCodeDSAScraper:
    def __init__(self, headless=True):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--user-agent=Mozilla/5.0")

        self.driver = None
        self.session = requests.Session()
        self.base_url = "https://leetcode.com/problems/"

    def start_driver(self):
        self.driver = webdriver.Chrome(options=self.options)

    def close_driver(self):
        if self.driver:
            self.driver.quit()

    def fetch_problem_list(self):
        print("Fetching problems list from LeetCode API...")
        url = "https://leetcode.com/api/problems/all/"
        try:
            res = self.session.get(url)
            data = res.json()
            return data['stat_status_pairs']
        except Exception as e:
            print(f"❌ Failed to fetch problems list: {e}")
            return []

    def filter_by_difficulty(self, problems, level):
        return [
            {
                "title": prob["stat"]["question__title"],
                "slug": prob["stat"]["question__title_slug"]
            }
            for prob in problems if prob["difficulty"]["level"] == level
        ]

    def scrape_problem_html(self, slug):
        url = self.base_url + slug
        print(f"Scraping: {url}")
        try:
            self.driver.get(url)
            time.sleep(5)  # wait for dynamic content to load
            html = self.driver.page_source
            return {
                "url": url,
                "slug": slug,
                "html": html
            }
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
            return None

    def run(self):
        self.start_driver()

        problems = self.fetch_problem_list()
        if not problems:
            print("No problems fetched.")
            return

        easy_list = self.filter_by_difficulty(problems, 1)
        medium_list = self.filter_by_difficulty(problems, 2)

        selected = random.sample(easy_list, 1) + random.sample(medium_list, 2)

        html_data = []
        for prob in selected:
            data = self.scrape_problem_html(prob["slug"])
            if data:
                html_data.append(data)

        self.close_driver()

        if html_data:
            out_dir = "app/dsa_coding/raw_html"
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "raw_problems.json"), "w", encoding="utf-8") as f:
                json.dump(html_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Saved {len(html_data)} problems to {out_dir}/raw_problems.json")
        else:
            print("❌ No problems saved.")

if __name__ == "__main__":
    scraper = LeetCodeDSAScraper(headless=False)
    scraper.run()
