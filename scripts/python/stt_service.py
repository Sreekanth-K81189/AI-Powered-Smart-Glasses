# Install deps: pip install -r requirements.txt (project root)
import sys
import os

def transcribe_wav(wav_path):
    """Transcribe a WAV file (16kHz mono). Used for push-to-talk."""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio, language="en-US")
            print(f"STT_RESULT:{text}", flush=True)
            return text
        except sr.UnknownValueError:
            print("STT_RESULT:", flush=True)
            return ""
        except sr.RequestError:
            print("STT_RESULT:", flush=True)
            return ""
    except Exception:
        print("STT_RESULT:", file=sys.stderr, flush=True)
        return ""

def listen(duration=7):
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8
        
        with sr.Microphone() as source:
            # Signal to C++ that we are listening
            print("STT_LISTENING", flush=True)
            r.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = r.listen(source, timeout=float(duration), phrase_time_limit=float(duration))
            except sr.WaitTimeoutError:
                print("STT_RESULT:", flush=True)
                return ""
        try:
            text = r.recognize_google(audio, language="en-US")
            print(f"STT_RESULT:{text}", flush=True)
            return text
        except sr.UnknownValueError:
            print("STT_RESULT:", flush=True)
            return ""
        except sr.RequestError as e:
            print(f"STT_RESULT:", flush=True)
            return ""
    except Exception as e:
        print(f"STT_RESULT:", file=sys.stderr, flush=True)
        return ""

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].lower().endswith(".wav"):
        transcribe_wav(sys.argv[1])  # already prints STT_RESULT: for C++
    else:
        dur = int(sys.argv[1]) if len(sys.argv) > 1 else 7
        result = listen(dur)
        if result:
            print(result, flush=True)
