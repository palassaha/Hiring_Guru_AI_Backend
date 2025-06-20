import os
from TTS.api import TTS

# Constants
USER_HOME = os.path.expanduser("~")
TTS_CACHE_DIR = os.path.join(USER_HOME, "AppData", "Local", "tts")
MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"

# Global variable to hold the loaded model
_tts_model = None

def is_model_downloaded(model_name: str, base_dir: str) -> bool:
    safe_name = model_name.replace("/", "--")
    return os.path.isdir(os.path.join(base_dir, safe_name))

def get_tts_model():
    global _tts_model
    if _tts_model is not None:
        return _tts_model

    if not is_model_downloaded(MODEL_NAME, TTS_CACHE_DIR):
        print(f"[⬇️] Downloading TTS model: {MODEL_NAME}...")

    print("[🧠] Loading TTS model into memory...")
    _tts_model = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)
    return _tts_model

def speak_text(text: str, output_path: str):
    try:
        print("[🔊] Synthesizing speech...")
        model = get_tts_model()
        model.tts_to_file(text=text, file_path=output_path)
        print(f"[💾] TTS saved at: {output_path}")
    except Exception as e:
        print(f"[❌] TTS synthesis failed: {e}")
