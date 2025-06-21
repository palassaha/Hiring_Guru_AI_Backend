import os
import json
import random
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI to use Groq's endpoint
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

class CommunicationQuestionGenerator:
    def __init__(self):
        """Initialize the generator - API key is set globally"""
        pass
        
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
        
        Return as a JSON array of sentences.
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
                sentences = [line.strip('- ').strip() for line in lines if line.strip()]
                return sentences[:count]
                
        except Exception as e:
            print(f"[❌] Error generating sentences: {e}")
            return []
    
    def generate_conversation_questions(self, count=15, category="general"):
        """Generate conversation starter questions"""
        categories = {
            "general": "everyday topics and general conversation",
            "personal": "personal experiences and preferences",
            "work": "professional and career-related topics",
            "hobbies": "interests, hobbies, and leisure activities",
            "travel": "travel experiences and places",
            "food": "food, cooking, and dining experiences"
        }
        
        topic = categories.get(category, "general conversation topics")
        
        system_prompt = "You are an English conversation teacher creating engaging discussion questions for students."
        
        user_prompt = f"""
        Generate {count} conversation starter questions about {topic}.
        
        Requirements:
        - Questions should be open-ended to encourage discussion
        - Use simple, clear language
        - Suitable for English learners
        - Mix of different question types (What, How, Why, Do you, Have you)
        - Encourage personal sharing and opinions
        
        Return as a JSON array of questions.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=1200
            )
            
            content = response['choices'][0]['message']['content'].strip()
            
            try:
                questions = json.loads(content)
                return questions
            except json.JSONDecodeError:
                lines = content.strip().split('\n')
                questions = [line.strip('- ').strip() for line in lines if line.strip() and '?' in line]
                return questions[:count]
                
        except Exception as e:
            print(f"[❌] Error generating conversation questions: {e}")
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
    
    def generate_comprehension_questions(self, passage, count=8):
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
        
        Format as JSON with this structure:
        {{
            "multiple_choice": [
                {{
                    "question": "question text",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A"
                }}
            ],
            "short_answer": [
                "question text"
            ]
        }}
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
    
    def generate_speaking_prompts(self, count=10, time_limit="2 minutes"):
        """Generate speaking prompts for communication practice"""
        system_prompt = "You are an English speaking coach creating practice prompts for students."
        
        user_prompt = f"""
        Generate {count} speaking prompts for English communication practice.
        
        Requirements:
        - Each prompt should be suitable for {time_limit} of speaking
        - Mix of descriptive, narrative, and opinion-based prompts
        - Use topics familiar to most people
        - Encourage personal expression
        - Include some with specific scenarios
        
        Return as a JSON array of prompts.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            
            content = response['choices'][0]['message']['content'].strip()
            
            try:
                prompts = json.loads(content)
                return prompts
            except json.JSONDecodeError:
                lines = content.strip().split('\n')
                prompts = [line.strip('- ').strip() for line in lines if line.strip()]
                return prompts[:count]
                
        except Exception as e:
            print(f"[❌] Error generating speaking prompts: {e}")
            return []

def main():
    # Check if API key is set
    if not openai.api_key:
        print("[❌] Please set your Groq API key in the GROQ_API_KEY environment variable")
        print("You can do this by creating a .env file with: GROQ_API_KEY=your_api_key_here")
        return
    
    generator = CommunicationQuestionGenerator()
    
    print("🎯 Communication Round Question Generator")
    print("=" * 50)
    
    while True:
        print("\nSelect an option:")
        print("1. Generate Basic English Sentences")
        print("2. Generate Conversation Questions")
        print("3. Generate Reading Comprehension")
        print("4. Generate Speaking Prompts")
        print("5. Generate Complete Question Set")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            try:
                count = int(input("Number of sentences (default 10): ") or 10)
                difficulty = input("Difficulty level (beginner/intermediate/advanced): ") or "beginner"
                
                print(f"\n📝 Generating {count} {difficulty} sentences...")
                sentences = generator.generate_basic_sentences(count, difficulty)
                
                print(f"\n✅ Basic English Sentences ({difficulty} level):")
                for i, sentence in enumerate(sentences, 1):
                    print(f"{i}. {sentence}")
            except ValueError:
                print("[❌] Please enter a valid number")
        
        elif choice == "2":
            try:
                count = int(input("Number of questions (default 15): ") or 15)
                category = input("Category (general/personal/work/hobbies/travel/food): ") or "general"
                
                print(f"\n💬 Generating {count} conversation questions...")
                questions = generator.generate_conversation_questions(count, category)
                
                print(f"\n✅ Conversation Questions ({category}):")
                for i, question in enumerate(questions, 1):
                    print(f"{i}. {question}")
            except ValueError:
                print("[❌] Please enter a valid number")
        
        elif choice == "3":
            topic = input("Passage topic (default: daily life): ") or "daily life"
            difficulty = input("Difficulty (beginner/intermediate/advanced): ") or "intermediate"
            
            print(f"\n📖 Generating reading comprehension...")
            passage = generator.generate_comprehension_passage(topic, difficulty)
            questions = generator.generate_comprehension_questions(passage)
            
            print(f"\n✅ Reading Comprehension - {topic.title()}:")
            print("\nPASSAGE:")
            print(passage)
            
            if questions.get("multiple_choice"):
                print("\nMULTIPLE CHOICE QUESTIONS:")
                for i, q in enumerate(questions.get("multiple_choice", []), 1):
                    print(f"\n{i}. {q['question']}")
                    for j, option in enumerate(q['options']):
                        print(f"   {chr(65+j)}. {option}")
                    print(f"   Answer: {q['correct_answer']}")
            
            if questions.get("short_answer"):
                print("\nSHORT ANSWER QUESTIONS:")
                for i, q in enumerate(questions.get("short_answer", []), 1):
                    print(f"{i}. {q}")
        
        elif choice == "4":
            try:
                count = int(input("Number of prompts (default 10): ") or 10)
                time_limit = input("Time limit (default: 2 minutes): ") or "2 minutes"
                
                print(f"\n🎤 Generating {count} speaking prompts...")
                prompts = generator.generate_speaking_prompts(count, time_limit)
                
                print(f"\n✅ Speaking Prompts ({time_limit} each):")
                for i, prompt in enumerate(prompts, 1):
                    print(f"{i}. {prompt}")
            except ValueError:
                print("[❌] Please enter a valid number")
        
        elif choice == "5":
            print("\n🎯 Generating complete question set...")
            
            # Generate all types
            sentences = generator.generate_basic_sentences(8, "intermediate")
            questions = generator.generate_conversation_questions(10, "general")
            passage = generator.generate_comprehension_passage("technology", "intermediate")
            comp_questions = generator.generate_comprehension_questions(passage, 6)
            prompts = generator.generate_speaking_prompts(5, "2 minutes")
            
            print("\n" + "="*60)
            print("COMPLETE COMMUNICATION ROUND QUESTION SET")
            print("="*60)
            
            print("\n📝 BASIC SENTENCES:")
            for i, sentence in enumerate(sentences, 1):
                print(f"{i}. {sentence}")
            
            print("\n💬 CONVERSATION STARTERS:")
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")
            
            print("\n📖 READING COMPREHENSION:")
            print(passage)
            
            if comp_questions.get("multiple_choice"):
                print("\nQUESTIONS:")
                for i, q in enumerate(comp_questions.get("multiple_choice", []), 1):
                    print(f"\n{i}. {q['question']}")
                    for j, option in enumerate(q['options']):
                        print(f"   {chr(65+j)}. {option}")
            
            print("\n🎤 SPEAKING PROMPTS:")
            for i, prompt in enumerate(prompts, 1):
                print(f"{i}. {prompt}")
        
        elif choice == "6":
            print("Goodbye!")
            break
        
        else:
            print("[❌] Invalid choice. Please try again.")

if __name__ == "__main__":
    main()