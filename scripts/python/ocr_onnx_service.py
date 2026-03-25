#!/usr/bin/env python3
"""
ocr_onnx_service.py — GPU-accelerated OCR using RapidOCR + ONNX Runtime CUDA
Called from C++ via PythonBridge (same pattern as ocr_service.py)

Models used (already downloaded):
  models/ocr_onnx/ocr_det.onnx  — text detection
  models/ocr_onnx/ocr_rec.onnx  — text recognition
  models/ocr_onnx/en_dict.txt   — character dictionary

Usage:
  python ocr_onnx_service.py <image_path>

Output:
  OCR_RESULT:<text>       on success
  OCR_EMPTY               if no text found
  OCR_ERROR:<message>     on error
"""

import sys
import os
import cv2

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MODEL_DIR    = os.path.join(PROJECT_ROOT, 'models', 'ocr_onnx')

DET_MODEL  = os.path.join(MODEL_DIR, 'ocr_det.onnx')
REC_MODEL  = os.path.join(MODEL_DIR, 'ocr_rec.onnx')
DICT_FILE  = os.path.join(MODEL_DIR, 'en_dict.txt')


def get_ocr_engine():
    """
    Initialize RapidOCR with ONNX Runtime GPU provider.
    RapidOCR uses ONNX Runtime CUDA under the hood.
    """
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR(
        det_model_path=DET_MODEL,
        rec_model_path=REC_MODEL,
        rec_keys_path=DICT_FILE,
        det_use_cuda=True,
        rec_use_cuda=True,
        det_gpu_id=0,
        rec_gpu_id=0,
    )
    return engine


_engine = None


def run_ocr(image_path: str) -> str:
    global _engine

    img = cv2.imread(image_path)
    if img is None:
        return f"OCR_ERROR:Cannot load image: {image_path}"

    try:
        if _engine is None:
            _engine = get_ocr_engine()

        # result: list of (bbox, text, confidence)
        result, elapse = _engine(img)

        if not result:
            return "OCR_EMPTY"

        lines = [item[1] for item in result if item[1].strip()]
        text  = " ".join(lines).strip()

        if not text:
            return "OCR_EMPTY"

        return f"OCR_RESULT:{text}"

    except Exception as e:
        return f"OCR_ERROR:{str(e)}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("OCR_ERROR:No image path provided")
        sys.exit(1)

    image_path = sys.argv[1]
    result = run_ocr(image_path)
    print(result)

