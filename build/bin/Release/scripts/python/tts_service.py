# Install deps: pip install -r requirements.txt (project root)
import sys, os

def speak(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 165)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        for v in voices:
            if 'english' in v.name.lower() or 'zira' in v.name.lower() or 'david' in v.name.lower():
                engine.setProperty('voice', v.id)
                break
        engine.say(text)
        engine.runAndWait()
        print("TTS_OK")
    except Exception as e:
        print(f"TTS_ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: tts_service.py <text>", file=sys.stderr)
        sys.exit(1)
    speak(" ".join(sys.argv[1:]))
