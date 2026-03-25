#!/bin/bash
# =============================================================================
# SIGN LANGUAGE TRANSLATION - INSTALLATION SCRIPT
# Run this in your Cursor terminal from your project root directory
# Requires: Python 3.8+, CUDA 11.8+, pip
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=================================================="
echo "  Sign Language Translation - Setup"
echo "=================================================="

# ---------------------------------------------------------------------------
# STEP 1: Install Python packages
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/5] Installing Python packages...${NC}"

pip install --upgrade pip setuptools wheel
pip install mediapipe==0.10.9
pip install numpy>=1.23.0
pip install opencv-python>=4.8.0
pip install onnxruntime-gpu==1.17.0
pip install protobuf==3.20.3
pip install scipy>=1.9.0

echo -e "${GREEN}✓ Python packages installed${NC}"

# ---------------------------------------------------------------------------
# STEP 2: Create directories
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/5] Creating directories...${NC}"

mkdir -p models/sign_language
mkdir -p scripts/python

echo -e "${GREEN}✓ Directories created${NC}"

# ---------------------------------------------------------------------------
# STEP 3: Download MediaPipe Hand Landmarker model (~29 MB)
# This detects 21 keypoints per hand in real-time
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/5] Downloading MediaPipe Hand Landmarker model...${NC}"

cd models/sign_language

if [ ! -f "hand_landmarker.task" ]; then
    wget -q --show-progress \
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
        -O hand_landmarker.task \
    || curl -L \
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
        -o hand_landmarker.task
    echo -e "${GREEN}✓ hand_landmarker.task downloaded${NC}"
else
    echo -e "${GREEN}✓ hand_landmarker.task already exists${NC}"
fi

# ---------------------------------------------------------------------------
# STEP 4: Download MediaPipe Gesture Recognizer model (~5 MB)
# This recognizes common ASL gestures and maps them to English words.
# Supports: hello, thank you, yes, no, stop, go, iloveyou, thumbs up/down etc.
# For extended word vocabulary, the LSTM in sign_language_service.py
# builds on top of the raw hand keypoints from hand_landmarker.task
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/5] Downloading ASL Gesture Recognizer model...${NC}"

if [ ! -f "gesture_recognizer.task" ]; then
    wget -q --show-progress \
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" \
        -O gesture_recognizer.task \
    || curl -L \
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" \
        -o gesture_recognizer.task
    echo -e "${GREEN}✓ gesture_recognizer.task downloaded${NC}"
else
    echo -e "${GREEN}✓ gesture_recognizer.task already exists${NC}"
fi

cd ../..

# ---------------------------------------------------------------------------
# STEP 5: Verify everything works
# ---------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/5] Verifying installation...${NC}"

python3 << 'PYEOF'
import sys

checks = []

try:
    import mediapipe as mp
    checks.append(("MediaPipe", mp.__version__, True))
except ImportError as e:
    checks.append(("MediaPipe", str(e), False))

try:
    import cv2
    checks.append(("OpenCV", cv2.__version__, True))
except ImportError as e:
    checks.append(("OpenCV", str(e), False))

try:
    import onnxruntime as ort
    cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    checks.append(("ONNX Runtime", f"{ort.__version__} | CUDA={'YES' if cuda else 'NO (CPU fallback)'}", True))
except ImportError as e:
    checks.append(("ONNX Runtime", str(e), False))

try:
    import numpy as np
    checks.append(("NumPy", np.__version__, True))
except ImportError as e:
    checks.append(("NumPy", str(e), False))

all_ok = True
for name, version, ok in checks:
    status = "✓" if ok else "✗"
    print(f"  {status} {name}: {version}")
    if not ok:
        all_ok = False

if not all_ok:
    print("\n⚠ Some packages failed. Check errors above.")
    sys.exit(1)
else:
    print("\n✓ All packages verified successfully")
PYEOF

echo ""
echo "=================================================="
echo -e "${GREEN}✓ INSTALLATION COMPLETE${NC}"
echo "=================================================="
echo ""
echo "Models downloaded to models/sign_language/:"
echo "  - hand_landmarker.task     (21 keypoint hand detection)"
echo "  - gesture_recognizer.task  (ASL word recognition)"
echo ""
echo "Next step: Copy these files into your project:"
echo "  scripts/python/sign_language_service.py"
echo "  src/SignLanguageTranslator.cpp"
echo "  include/SignLanguageTranslator.hpp"
echo "=================================================="
