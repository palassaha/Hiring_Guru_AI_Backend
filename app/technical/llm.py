
import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -- ✅ Configure Groq API
openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"

def generate_answer_groq(question, options):
    formatted_options = "\n".join([f"- {opt}" for opt in options])

    prompt = f"""
You are an exam evaluator. Your task is to choose the correct answer to the multiple-choice question below. Select the correct answer **only** from the given options. 

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
        print(f"❌ Error for question: {question[:50]}... -> {e}")
        return ""

def process_questions(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        question = q["question"]
        options = q["options"]
        answer = generate_answer_groq(question, options)
        q["answer"] = answer
        q["explanation"] = ""
        print(f"✅ {question}\n➡️  {answer}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=4)

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, "question_bank.json")
    output_path = os.path.join(base_dir, "questions_with_answers.json")

    print(f"📂 Reading: {input_path}")
    print(f"💾 Writing: {output_path}\n")
    process_questions(input_path, output_path)
