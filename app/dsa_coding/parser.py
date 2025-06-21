import os
import json
import re
import sys

# Ensure root path is included in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.config import GROQ_API_KEY
import openai

openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"

RAW_PROBLEMS_FILE = "app/dsa_coding/raw_html/raw_problems.json"
PARSED_OUTPUT_FILE = "app/dsa_coding/questions.json"

def extract_json_from_text(text):
    """Try to extract JSON object from arbitrary LLM output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text  # fallback

def build_prompt(html):
    # Trim HTML length to avoid token limits - adjust as needed
    trimmed_html = html[:3000]  # sending first 3000 chars only; adjust this to your token budget

    prompt = (
        "You are a helpful assistant that extracts detailed programming problem data from raw HTML content.\n"
        "Extract the following fields and respond ONLY with a JSON object containing:\n"
        "  - title\n"
        "  - problem_statement\n"
        "  - constraints\n"
        "  - input_format\n"
        "  - output_format\n"
        "  - examples (list of dicts with input/output keys)\n"
        "Use the following HTML content to extract this info:\n\n"
        f"{trimmed_html}\n\n"
        "If some field is missing, return an empty string or empty list.\n"
        "Respond with valid JSON only, no explanations."
    )
    return prompt

def parse_with_llm(html):
    prompt = build_prompt(html)
    try:
        response = openai.ChatCompletion.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful programming problem parser. Respond ONLY with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        content = response['choices'][0]['message']['content']
        print("\n[LLM Raw Output]:\n", content)  # debug print

        json_str = extract_json_from_text(content)
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error parsing with LLM: {e}")
        return None

def main():
    if not os.path.exists(RAW_PROBLEMS_FILE):
        print(f"Raw problems file not found: {RAW_PROBLEMS_FILE}")
        return

    with open(RAW_PROBLEMS_FILE, "r", encoding="utf-8") as f:
        raw_problems = json.load(f)

    parsed_problems = []
    for i, prob in enumerate(raw_problems, 1):
        print(f"\n🔍 Parsing problem {i}: {prob.get('url', 'N/A')}")
        html = prob.get("html", "")
        if not html:
            print("❌ No HTML content, skipping.")
            continue

        parsed = parse_with_llm(html)
        if parsed:
            # Add URL and difficulty from raw if available
            parsed['url'] = prob.get('url', '')
            parsed['difficulty'] = prob.get('difficulty', '')
            parsed_problems.append(parsed)
            print(f"✅ Parsed problem {i}")
        else:
            print(f"❌ Failed to parse problem {i}")

    os.makedirs(os.path.dirname(PARSED_OUTPUT_FILE), exist_ok=True)
    with open(PARSED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(parsed_problems, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(parsed_problems)} parsed problems to {PARSED_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
