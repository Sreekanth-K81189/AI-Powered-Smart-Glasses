#!/usr/bin/env python3
"""
sign_onnx_service.py — GPU-accelerated sign language recognition
Called from C++ via PythonBridge (same pattern as stt_service.py)

Uses:
  - MediaPipe HandLandmarker  (hand detection — already fast)
  - MediaPipe GestureRecognizer (gesture classification)

Models used (already downloaded):
  models/sign_language/gesture_recognizer.task
  models/sign_language/hand_landmarker.task

Usage:
  python sign_onnx_service.py <image_path>

Output:
  SIGN_RESULT:<word>:<confidence>
  SIGN_EMPTY
  SIGN_ERROR:<message>
"""

import sys
import os
import cv2
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MODEL_DIR    = os.path.join(PROJECT_ROOT, 'models', 'sign_language')

GESTURE_MODEL = os.path.join(MODEL_DIR, 'gesture_recognizer.task')
HAND_MODEL    = os.path.join(MODEL_DIR, 'hand_landmarker.task')

MIN_CONFIDENCE = 0.6

GESTURE_TO_ENGLISH = {
    "None":           "",
    "Closed_Fist":    "stop",
    "Open_Palm":      "hello",
    "Pointing_Up":    "one",
    "Thumb_Down":     "no",
    "Thumb_Up":       "yes",
    "Victory":        "peace",
    "ILoveYou":       "i love you",
}

_recognizer = None


def get_recognizer():
    """
    Singleton MediaPipe GestureRecognizer instance.
    """
    global _recognizer
    if _recognizer is not None:
        return _recognizer

    if not os.path.exists(GESTURE_MODEL):
        raise FileNotFoundError(f"Model not found: {GESTURE_MODEL}")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=GESTURE_MODEL)
    options = mp_vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    _recognizer = mp_vision.GestureRecognizer.create_from_options(options)
    return _recognizer


def run_sign(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return f"SIGN_ERROR:Cannot load image: {image_path}"

    try:
        import mediapipe as mp
        recognizer = get_recognizer()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)

        if not result.gestures:
            return "SIGN_EMPTY"

        top        = result.gestures[0][0]
        confidence = top.score
        raw_name   = top.category_name

        if confidence < MIN_CONFIDENCE:
            return "SIGN_EMPTY"

        english = GESTURE_TO_ENGLISH.get(raw_name, "")
        if not english:
            return "SIGN_EMPTY"

        return f"SIGN_RESULT:{english}:{confidence:.2f}"

    except Exception as e:
        return f"SIGN_ERROR:{str(e)}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("SIGN_ERROR:No image path provided")
        sys.exit(1)

    result = run_sign(sys.argv[1])
    print(result)

