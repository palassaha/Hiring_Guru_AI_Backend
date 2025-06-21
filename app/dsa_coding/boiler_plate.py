import json
import os


async def generate_boilerplates_async(input_file: str) -> list:
    """
    Async wrapper for the boilerplate generation logic
    """
    import asyncio
    import base64
    import re
    import openai
    from dotenv import load_dotenv
    
    # Load Groq API Key from .env
    load_dotenv()
    openai.api_key = os.getenv("GROQ_API_KEY")
    openai.api_base = "https://api.groq.com/openai/v1"
    
    def clean_code_output(raw_code: str) -> str:
        lines = raw_code.strip().splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(lines).strip()

    def encode_base64(text: str) -> str:
        return base64.b64encode(text.encode("utf-8")).decode("utf-8")

    def generate_code_template(problem_description: str, language: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code generation assistant. Given a problem and a programming language, "
                    "generate a code template that:\n"
                    "1. Reads input from stdin\n"
                    "2. Prints output to stdout\n"
                    "3. Has basic structure (e.g. main function, imports)\n"
                    "4. Includes a '// TODO' or '# Your code here' comment\n"
                    "5. Does NOT solve the problem — just the skeleton.\n"
                    "Only output the code. Do not explain anything. No Markdown formatting."
                )
            },
            {
                "role": "user",
                "content": f"Language: {language}\nProblem: {problem_description}"
            }
        ]

        response = openai.ChatCompletion.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=512
        )

        raw_code = response.choices[0].message.content.strip()
        return clean_code_output(raw_code)

    def extract_problem_text(problem_data: dict) -> str:
        title = problem_data.get("title", "No Title")
        statement = problem_data.get("problem_statement", "")
        input_format = problem_data.get("input_format", "")
        constraints = "\n".join(problem_data.get("constraints", []))
        return f"{title}\n\n{statement}\n\nInput Format:\n{input_format}\n\nConstraints:\n{constraints}"

    def load_all_problems(filepath: str) -> list:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def transform_input(raw_input: str) -> str:
        """
        Extract all occurrences of '[...]' assigned to variables in the input string,
        regardless of commas inside the brackets.
        """
        pattern = r"\w+\s*=\s*(\[[^\]]*\])"
        matches = re.findall(pattern, raw_input)
        return " ".join(matches)

    # Main processing logic
    languages = ["Python", "Java", "C++"]
    problems = load_all_problems(input_file)
    structured_output = []

    for prob in problems:
        prob_id = prob.get("frontend_id") or prob.get("question_id") or prob.get("title_slug") or "unknown_id"
        print(f"\nGenerating templates for problem: {prob_id}")
        desc = extract_problem_text(prob)
        boilerplates = {}

        for lang in languages:
            print(f"  - {lang}")
            try:
                template = generate_code_template(desc, lang)
                encoded_template = encode_base64(template)
                boilerplates[lang] = encoded_template
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            except Exception as e:
                boilerplates[lang] = f"ERROR: {str(e)}"

        test_cases = []
        for example in prob.get("examples", []):
            try:
                raw_input = example.get("input", "")
                if not raw_input:
                    print(f"⚠️ Warning: Empty input in example for problem {prob.get('title', '')}")
                transformed_input = transform_input(str(raw_input).strip())
                output_text = str(example.get("output", "")).strip()
                test_cases.append({
                    "input": transformed_input,
                    "output": output_text
                })
            except Exception as e:
                print(f"❌ Failed to process example: {example}\nError: {e}")
                test_cases.append({
                    "input": "ERROR",
                    "output": "ERROR"
                })

        structured_output.append({
            "Problem Statement": prob.get("problem_statement", ""),
            "Title": prob.get("title", ""),
            "test cases": test_cases,
            "Boiler Plate": boilerplates
        })

    return structured_output