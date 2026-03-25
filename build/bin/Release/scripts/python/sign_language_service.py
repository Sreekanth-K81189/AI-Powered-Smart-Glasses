#!/usr/bin/env python3
"""
Sign Language Service - ONNX Runtime + CUDA accelerated
Called from C++ via PythonBridge.

Usage:
    python sign_language_service.py <image_path>
    python sign_language_service.py --camera <duration>

Output:
    SIGN_RESULT:<word>:<confidence>
    SIGN_EMPTY
    SIGN_ERROR:<message>
"""

import sys
import os
import cv2
import numpy as np
import argparse
import time

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MODEL_DIR    = os.path.join(PROJECT_ROOT, 'models', 'sign_language')

# Gesture label map
GESTURE_LABELS = {
    0: "none",
    1: "closed_fist",
    2: "open_palm",
    3: "pointing_up",
    4: "thumbs_down",
    5: "thumbs_up",
    6: "victory",
    7: "i_love_you",
}

GESTURE_TO_ENGLISH = {
    "none":        "",
    "closed_fist": "stop",
    "open_palm":   "hello",
    "pointing_up": "one",
    "thumbs_down": "no",
    "thumbs_up":   "yes",
    "victory":     "peace",
    "i_love_you":  "i love you",
}

MIN_CONFIDENCE = 0.6


def get_ort_session(model_path: str):
    """
    Create ONNX Runtime session with CUDA provider if available,
    falling back to CPU. This is what gives GPU acceleration.
    """
    import onnxruntime as ort

    providers = ort.get_available_providers()
    cuda_available = 'CUDAExecutionProvider' in providers

    if cuda_available:
        provider_list = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
    else:
        provider_list = ['CPUExecutionProvider']

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    session = ort.InferenceSession(model_path, sess_options=opts,
                                   providers=provider_list)

    active = session.get_providers()
    if 'CUDAExecutionProvider' in active:
        print(f"[SignLang] Using CUDA GPU acceleration", file=sys.stderr)
    else:
        print(f"[SignLang] Using CPU (CUDA not available)", file=sys.stderr)

    return session


def extract_hand_keypoints_mediapipe(frame_bgr: np.ndarray):
    """
    Extract 21 hand landmarks using MediaPipe.
    Returns flat array of 63 floats (21 * xyz) or None if no hand detected.
    """
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            model_complexity=0   # 0=lite for speed
        ) as hands:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if not result.multi_hand_landmarks:
                return None
            lms = result.multi_hand_landmarks[0]
            kp = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark],
                          dtype=np.float32)
            # Normalize: translate to wrist, scale by hand span
            kp -= kp[0]
            scale = np.linalg.norm(kp[9])
            if scale > 0:
                kp /= scale
            return kp.flatten()
    except Exception as e:
        print(f"[SignLang] MediaPipe error: {e}", file=sys.stderr)
        return None


def recognize_with_mediapipe_gesture(frame_bgr: np.ndarray):
    """
    Use MediaPipe GestureRecognizer as primary recognizer.
    Returns (word, confidence) or ("", 0.0)
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = os.path.join(MODEL_DIR, 'gesture_recognizer.task')
        if not os.path.exists(model_path):
            return "", 0.0

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        recognizer = mp_vision.GestureRecognizer.create_from_options(options)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)

        if not result.gestures:
            return "", 0.0

        top = result.gestures[0][0]
        confidence = top.score
        raw_name   = top.category_name

        if confidence < MIN_CONFIDENCE:
            return "", confidence

        english = GESTURE_TO_ENGLISH.get(raw_name.lower(), "")
        return english, confidence

    except Exception as e:
        print(f"[SignLang] Gesture recognizer error: {e}", file=sys.stderr)
        return "", 0.0


def recognize_from_image(image_path: str) -> str:
    """
    Main recognition function. Tries MediaPipe GestureRecognizer first.
    Returns SIGN_RESULT:<word>:<confidence> or SIGN_EMPTY or SIGN_ERROR:...
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return f"SIGN_ERROR:Cannot load image: {image_path}"

    try:
        word, confidence = recognize_with_mediapipe_gesture(frame)

        if word and confidence >= MIN_CONFIDENCE:
            return f"SIGN_RESULT:{word}:{confidence:.2f}"
        else:
            return "SIGN_EMPTY"

    except Exception as e:
        return f"SIGN_ERROR:{str(e)}"


def recognize_from_camera(duration_seconds: int = 3) -> str:
    """
    Camera-based recognition for N seconds. Returns best result.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "SIGN_ERROR:Cannot open camera"

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    best_word = ""
    best_conf = 0.0
    start = time.time()

    try:
        while (time.time() - start) < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            word, conf = recognize_with_mediapipe_gesture(frame)
            if word and conf > best_conf:
                best_word = word
                best_conf = conf
    finally:
        cap.release()

    if best_word:
        return f"SIGN_RESULT:{best_word}:{best_conf:.2f}"
    return "SIGN_EMPTY"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Service")
    parser.add_argument("input",     nargs="?", help="Image path")
    parser.add_argument("--camera",  type=int,  help="Camera duration in seconds")
    args = parser.parse_args()

    if args.camera is not None:
        print(recognize_from_camera(args.camera))
    elif args.input:
        print(recognize_from_image(args.input))
    else:
        print("SIGN_ERROR:No input. Use: sign_language_service.py <image_path>")
        sys.exit(1)

