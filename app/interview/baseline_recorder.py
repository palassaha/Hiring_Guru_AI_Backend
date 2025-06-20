
import sounddevice as sd
import scipy.io.wavfile as wav
import json
from datetime import datetime
from app.interview.feature_extract import extract_voice_features

def record_baseline_audio(filename="baseline.wav", duration=10, fs=16000):
    """
    Records Whisper-compatible mono 16kHz audio as baseline.
    """
    print(f"[🎙️] Recording calm voice for {duration} seconds at {fs} Hz (mono)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print(f"[💾] Baseline audio saved to: {filename}")
    return filename

def create_baseline_file():
    try:
        audio_path = record_baseline_audio()

        print("[🔍] Extracting features from baseline audio...")
        features = extract_voice_features(audio_path)
        print("[📈] Baseline Features:", features)

        with open("baseline_features.json", "w") as f:
            json.dump(features, f, indent=2)

        print("[✅] Baseline features saved to: baseline_features.json")
    except Exception as e:
        print(f"[❌] Failed to create baseline: {e}")

if __name__ == "__main__":
    create_baseline_file()
