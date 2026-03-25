# download_cuda_models.py
# Run from your project root:
#   python download_cuda_models.py
#
# Downloads:
#   - PaddleOCR ONNX models (det + rec + dict) from HuggingFace
#   - Hand gesture ONNX model from HuggingFace
# All models work with ONNX Runtime + CUDA

import os
import sys
import urllib.request
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OCR_DIR      = os.path.join(PROJECT_ROOT, "models", "ocr_onnx")
SIGN_DIR     = os.path.join(PROJECT_ROOT, "models", "sign_language")

os.makedirs(OCR_DIR,  exist_ok=True)
os.makedirs(SIGN_DIR, exist_ok=True)

def progress(count, block_size, total_size):
    pct = min(int(count * block_size * 100 / total_size), 100)
    bar = "#" * (pct // 2)
    print(f"\r  [{bar:<50}] {pct}%", end="", flush=True)

def download(url, dest, label):
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024*1024)
        print(f"  SKIP  {label} already exists ({size_mb:.1f} MB)")
        return True
    print(f"  Downloading {label}...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()
        size_mb = os.path.getsize(dest) / (1024*1024)
        print(f"  PASS  {label} downloaded ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  FAIL  {label}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False

print()
print("=" * 55)
print("  CUDA Model Downloader")
print("=" * 55)

# Install huggingface_hub if needed
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("\nInstalling huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import hf_hub_download

print()
print("[1/2] Downloading PaddleOCR ONNX models from HuggingFace...")
print("      Source: monkt/paddleocr-onnx (Apache 2.0)")
print()

ocr_ok = True

# Detection model
det_dest = os.path.join(OCR_DIR, "ocr_det.onnx")
if os.path.exists(det_dest):
    print(f"  SKIP  ocr_det.onnx already exists")
else:
    try:
        print("  Downloading ocr_det.onnx (~2 MB)...")
        path = hf_hub_download(
            repo_id="monkt/paddleocr-onnx",
            filename="detection/v5/det.onnx",
            local_dir=OCR_DIR
        )
        shutil.copy(path, det_dest)
        print(f"  PASS  ocr_det.onnx downloaded")
    except Exception as e:
        print(f"  FAIL  ocr_det.onnx: {e}")
        ocr_ok = False

# Recognition model
rec_dest = os.path.join(OCR_DIR, "ocr_rec.onnx")
if os.path.exists(rec_dest):
    print(f"  SKIP  ocr_rec.onnx already exists")
else:
    try:
        print("  Downloading ocr_rec.onnx (~8 MB)...")
        path = hf_hub_download(
            repo_id="monkt/paddleocr-onnx",
            filename="languages/english/rec.onnx",
            local_dir=OCR_DIR
        )
        shutil.copy(path, rec_dest)
        print(f"  PASS  ocr_rec.onnx downloaded")
    except Exception as e:
        print(f"  FAIL  ocr_rec.onnx: {e}")
        ocr_ok = False

# Dictionary
dict_dest = os.path.join(OCR_DIR, "en_dict.txt")
if os.path.exists(dict_dest):
    print(f"  SKIP  en_dict.txt already exists")
else:
    try:
        print("  Downloading en_dict.txt...")
        path = hf_hub_download(
            repo_id="monkt/paddleocr-onnx",
            filename="languages/english/dict.txt",
            local_dir=OCR_DIR
        )
        shutil.copy(path, dict_dest)
        print(f"  PASS  en_dict.txt downloaded")
    except Exception as e:
        print(f"  FAIL  en_dict.txt: {e}")
        ocr_ok = False

print()
print("[2/2] Downloading Hand Gesture ONNX model from HuggingFace...")
print("      Source: Ultralytics/assets (gesture classifier)")
print()

sign_ok = True
gesture_dest = os.path.join(SIGN_DIR, "gesture_model.onnx")

if os.path.exists(gesture_dest):
    print(f"  SKIP  gesture_model.onnx already exists")
else:
    # Try multiple sources in order
    gesture_sources = [
        # Source 1: Ultralytics hand gesture classifier (direct URL)
        (
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobilenet_v3_small_100_224-224-3-gesture.onnx",
            "Ultralytics MobileNetV3 gesture model"
        ),
        # Source 2: direct ONNX from HuggingFace spaces
        (
            "https://huggingface.co/Ultralytics/assets/resolve/main/hand-gesture-onnx/model.onnx",
            "HuggingFace gesture model"
        ),
    ]

    downloaded = False
    for url, label in gesture_sources:
        print(f"  Trying: {label}...")
        ok = download(url, gesture_dest, "gesture_model.onnx")
        if ok:
            downloaded = True
            break

    if not downloaded:
        # Fallback: use Python to generate a simple gesture model via mediapipe
        print("  INFO  Direct download failed. Using MediaPipe gesture_recognizer.task instead.")
        print("        (Already downloaded during fix_sign_language.ps1)")
        gesture_task = os.path.join(SIGN_DIR, "gesture_recognizer.task")
        if os.path.exists(gesture_task):
            print(f"  PASS  gesture_recognizer.task found — sign language will use this")
            sign_ok = True
        else:
            print(f"  FAIL  gesture_recognizer.task also missing")
            print(f"        Re-run fix_sign_language.ps1 to download it")
            sign_ok = False

# Also install rapidocr for easier ONNX OCR pipeline
print()
print("[3/3] Installing rapidocr_onnxruntime (handles ONNX OCR pipeline)...")
ret = os.system(f"{sys.executable} -m pip install rapidocr_onnxruntime -q")
if ret == 0:
    print("  PASS  rapidocr_onnxruntime installed")
else:
    print("  WARN  rapidocr_onnxruntime install failed (optional)")

# Final verification
print()
print("=" * 55)
print("  Verification")
print("=" * 55)
print()

checks = [
    (os.path.join(OCR_DIR,  "ocr_det.onnx"),            "OCR detection model"),
    (os.path.join(OCR_DIR,  "ocr_rec.onnx"),            "OCR recognition model"),
    (os.path.join(OCR_DIR,  "en_dict.txt"),             "OCR dictionary"),
    (os.path.join(SIGN_DIR, "gesture_model.onnx"),      "Gesture ONNX model"),
    (os.path.join(SIGN_DIR, "gesture_recognizer.task"), "Gesture MediaPipe model"),
    (os.path.join(SIGN_DIR, "hand_landmarker.task"),    "Hand landmarker model"),
]

passed = 0
for path, label in checks:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"  [PASS] {label} ({size_mb:.1f} MB)")
        passed += 1
    else:
        print(f"  [MISS] {label} — {path}")

print()
print(f"  {passed}/{len(checks)} files present")
print()

if passed >= 4:
    print("  READY — Enough models present to run CUDA OCR + Sign Language")
    print()
    print("  Next step: paste CURSOR_FULL_CUDA_INTEGRATION.md into Cursor")
else:
    print("  INCOMPLETE — Check your internet connection and re-run")

print("=" * 55)
print()
