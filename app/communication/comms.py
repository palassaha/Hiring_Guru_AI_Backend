import json

import openai


class CommunicationQuestionGenerator:
    def __init__(self):
        """Initialize the generator - API key is set globally"""
        if not openai.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
    def generate_basic_sentences(self, count=10, difficulty="beginner"):
        """Generate basic English sentences for practice"""
        system_prompt = "You are an English language teacher creating practice sentences for students."
        
        user_prompt = f"""
        Generate {count} simple {difficulty}-level English sentences for communication practice.
        
        Requirements:
        - Use common vocabulary
        - Include different sentence types (statements, questions, commands)
        - Cover daily life topics (family, work, hobbies, food, weather)
        - Make them suitable for speaking practice
        
        Return as a JSON array of sentences only, no additional text.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = response['choices'][0]['message']['content'].strip()
            
            # Try to parse JSON, fallback to text processing if needed
            try:
                sentences = json.loads(content)
                return sentences
            except json.JSONDecodeError:
                # Extract sentences from text response
                lines = content.strip().split('\n')
                sentences = [line.strip('- ').strip().strip('"') for line in lines if line.strip()]
                return sentences[:count]
                
        except Exception as e:
            print(f"[❌] Error generating sentences: {e}")
            return []
    
    def generate_comprehension_passage(self, topic="daily life", difficulty="intermediate"):
        """Generate a reading comprehension passage"""
        system_prompt = "You are an English teacher creating reading comprehension materials for students."
        
        user_prompt = f"""
        Write a {difficulty}-level English reading comprehension passage about {topic}.
        
        Requirements:
        - 150-200 words long
        - Clear, simple language appropriate for {difficulty} level
        - Interesting and engaging content
        - Include specific details that can be questioned
        - Use proper paragraphs
        
        Topic: {topic}
        Return only the passage text, no additional formatting.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                max_tokens=800
            )
            
            return response['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            print(f"[❌] Error generating passage: {e}")
            return ""
    
    def generate_comprehension_questions(self, passage, count=5):
        """Generate questions based on a reading passage"""
        system_prompt = "You are an English teacher creating comprehension questions for a reading passage."
        
        user_prompt = f"""
        Based on the following passage, create {count} comprehension questions.
        
        Passage:
        {passage}
        
        Requirements:
        - Mix of question types: factual, inferential, vocabulary, and opinion
        - Include multiple choice (4 options each) and short answer questions
        - Questions should test understanding of main ideas and details
        - Use clear, simple language
        
        Format as JSON with this exact structure:
        {{
            "multiple_choice": [
                {{
                    "question": "question text",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "A"
                }}
            ],
            "short_answer": [
                "question text"
            ]
        }}
        
        Return only the JSON, no additional text.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            content = response['choices'][0]['message']['content'].strip()
            
            try:
                questions = json.loads(content)
                return questions
            except json.JSONDecodeError:
                print("[❌] Could not parse questions as JSON")
                return {"multiple_choice": [], "short_answer": []}
                
        except Exception as e:
            print(f"[❌] Error generating comprehension questions: {e}")
            return {"multiple_choice": [], "short_answer": []}
