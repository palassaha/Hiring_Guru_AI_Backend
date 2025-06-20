from difflib import SequenceMatcher
import re

class PronunciationScorer:
   def __init__(self):
       pass
   
   def normalize_text(self, text):
       """Normalize text for comparison"""
       # Remove punctuation and convert to lowercase
       text = re.sub(r'[^\w\s]', '', text.lower())
       # Remove extra whitespace
       text = ' '.join(text.split())
       return text
   
   def calculate_similarity(self, original_text, spoken_text):
       """Calculate similarity percentage between original and spoken text"""
       # Normalize both texts
       original_normalized = self.normalize_text(original_text)
       spoken_normalized = self.normalize_text(spoken_text)
       
       # Calculate similarity using SequenceMatcher
       similarity = SequenceMatcher(None, original_normalized, spoken_normalized).ratio()
       
       # Convert to percentage
       percentage = round(similarity * 100, 2)
       
       return {
           "similarity_percentage": percentage,
           "original_text": original_text,
           "spoken_text": spoken_text,
           "normalized_original": original_normalized,
           "normalized_spoken": spoken_normalized
       }
