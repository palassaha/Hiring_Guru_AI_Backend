
import cv2
import sounddevice as sd
import soundfile as sf
import threading
import os
from datetime import datetime

def create_session_folder():
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    path = os.path.join("data", "interviews", session_id)
    os.makedirs(path, exist_ok=True)
    return path

def record_audio(file_path, duration=10, fs=16000):
    """
    Records mono, 16-bit PCM audio at 16kHz (Whisper-compatible).
    """
    print(f"[🎙️ Audio] Recording for {duration} seconds at {fs} Hz (mono)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    sf.write(file_path, audio, fs, subtype='PCM_16')
    print(f"[💾 Audio] Saved to {file_path}")


def record_video(path, duration=10):
    print("[🎥 Video] Recording...")
    cap = cv2.VideoCapture(0)
    width = int(cap.get(3))
    height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(path, "user_video.mp4")
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[💾 Video] Saved to {video_path}")

def record_audio_video(duration=10):
    session_path = create_session_folder()

    # Run audio + video recording in parallel
    audio_thread = threading.Thread(target=record_audio, args=(session_path, duration))
    video_thread = threading.Thread(target=record_video, args=(session_path, duration))

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()

    return session_path
