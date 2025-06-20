
from playsound import playsound

def play_audio(file_path: str):
    try:
        print(f"[🔈] Playing audio: {file_path}")
        playsound(file_path)
    except Exception as e:
        print(f"[❌] Audio playback failed: {e}")
