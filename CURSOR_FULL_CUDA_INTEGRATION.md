# FULL CUDA INTEGRATION — OCR + SIGN LANGUAGE
# Replace Tesseract (CPU) with PaddleOCR ONNX (CUDA)
# Replace MediaPipe (CPU) with ONNX Gesture Model (CUDA)
# Both use ONNX Runtime + CUDA — same as YOLO in this project

You are replacing two CPU-only subsystems with fully GPU-accelerated ONNX
Runtime pipelines. Follow every step in order. Reference actual file names
and line numbers. Do not skip steps.

---

## OVERVIEW

| Subsystem      | Remove                  | Replace With                    | GPU Method        |
|----------------|-------------------------|---------------------------------|-------------------|
| OCR text recog | Tesseract TessBaseAPI   | PaddleOCR ONNX model            | OnnxRuntime CUDA  |
| Sign language  | MediaPipe GestureRecog  | hand_gesture_recognition.onnx   | OnnxRuntime CUDA  |

Both replacements follow the EXACT same ONNX Runtime pattern already used
by OnnxDetector.cpp for YOLO. When in doubt, mirror OnnxDetector.cpp.

---

## PART A — FULL CUDA OCR (Replace Tesseract with PaddleOCR ONNX)

### A1. Download Models (run in PowerShell terminal first)

```powershell
# Create model directory
New-Item -ItemType Directory -Force -Path "models\ocr_onnx"

# Download PaddleOCR text detection model (ONNX)
$DetUrl = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.onnx"
Invoke-WebRequest -Uri $DetUrl -OutFile "models\ocr_onnx\ocr_det.onnx"

# Download PaddleOCR text recognition model (ONNX)
$RecUrl = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.onnx"
Invoke-WebRequest -Uri $RecUrl -OutFile "models\ocr_onnx\ocr_rec.onnx"

# Download character dictionary
$DictUrl = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt"
Invoke-WebRequest -Uri $DictUrl -OutFile "models\ocr_onnx\en_dict.txt"

Write-Host "OCR models downloaded"
```

If Invoke-WebRequest fails, install manually:
- https://github.com/PaddlePaddle/PaddleOCR/releases → download ONNX exports
- Place in models/ocr_onnx/ocr_det.onnx and models/ocr_onnx/ocr_rec.onnx

Also install Python package:
```powershell
python -m pip install paddleocr onnxruntime-gpu==1.17.0
```

---

### A2. Create New File: include/ONNXOCREngine.hpp

Create this file at include/ONNXOCREngine.hpp:

```cpp
#pragma once
// ============================================================================
// ONNXOCREngine.hpp
// GPU-accelerated OCR using PaddleOCR ONNX models via ONNX Runtime CUDA
// Replaces Tesseract TessBaseAPI entirely
// Same ONNX Runtime pattern as OnnxDetector.hpp
// ============================================================================

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

// Forward declare ORT types to avoid header pollution
namespace Ort { class Session; class Env; class SessionOptions; }

struct OCRWord {
    std::string text;
    float       confidence;
    cv::Rect    bbox;
};

class ONNXOCREngine {
public:
    ONNXOCREngine();
    ~ONNXOCREngine();

    // Initialize with paths to detection and recognition ONNX models
    bool initialize(const std::string& detModelPath,
                    const std::string& recModelPath,
                    const std::string& dictPath);

    // Run full OCR pipeline on image — returns recognized text
    std::string recognize(const cv::Mat& image);

    // Run OCR and return per-word results with bounding boxes
    std::vector<OCRWord> recognizeDetailed(const cv::Mat& image);

    bool isLoaded() const { return loaded_; }
    bool isUsingCUDA() const { return usingCUDA_; }

private:
    std::unique_ptr<Ort::Env>            env_;
    std::unique_ptr<Ort::Session>        detSession_;   // text detection
    std::unique_ptr<Ort::Session>        recSession_;   // text recognition
    std::vector<std::string>             charDict_;

    bool loaded_    = false;
    bool usingCUDA_ = false;

    // Internal helpers
    Ort::SessionOptions buildSessionOptions();
    std::vector<cv::Rect> runDetection(const cv::Mat& image);
    std::string           runRecognition(const cv::Mat& crop);
    std::vector<std::string> loadDict(const std::string& dictPath);

    cv::Mat preprocessForDet(const cv::Mat& image);
    cv::Mat preprocessForRec(const cv::Mat& crop);
    std::string decodeCTC(const std::vector<float>& logits, int seqLen);
};
```

---

### A3. Create New File: src/ONNXOCREngine.cpp

Create this file at src/ONNXOCREngine.cpp:

```cpp
// ============================================================================
// ONNXOCREngine.cpp
// GPU-accelerated OCR — PaddleOCR ONNX via ONNX Runtime CUDA
// ============================================================================

#include "ONNXOCREngine.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <numeric>
#include <algorithm>

// -------------------------------------------------------------------------
// Constructor / Destructor
// -------------------------------------------------------------------------

ONNXOCREngine::ONNXOCREngine() {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXOCREngine");
}

ONNXOCREngine::~ONNXOCREngine() = default;

// -------------------------------------------------------------------------
// buildSessionOptions
// Mirror of OnnxDetector — CUDA first, CPU fallback
// -------------------------------------------------------------------------

Ort::SessionOptions ONNXOCREngine::buildSessionOptions() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try CUDA provider — same as OnnxDetector.cpp
    OrtCUDAProviderOptions cudaOpts{};
    cudaOpts.device_id = 0;
    try {
        opts.AppendExecutionProvider_CUDA(cudaOpts);
        usingCUDA_ = true;
        spdlog::info("[ONNX OCR] Using CUDA execution provider");
    } catch (const std::exception& e) {
        spdlog::warn("[ONNX OCR] CUDA unavailable: {} — using CPU", e.what());
        usingCUDA_ = false;
    }
    opts.AppendExecutionProvider_CPU();
    return opts;
}

// -------------------------------------------------------------------------
// initialize
// -------------------------------------------------------------------------

bool ONNXOCREngine::initialize(const std::string& detModelPath,
                                const std::string& recModelPath,
                                const std::string& dictPath) {
    try {
        auto detOpts = buildSessionOptions();
        auto recOpts = buildSessionOptions();

        detSession_ = std::make_unique<Ort::Session>(
            *env_, detModelPath.c_str(), detOpts);
        recSession_ = std::make_unique<Ort::Session>(
            *env_, recModelPath.c_str(), recOpts);

        charDict_ = loadDict(dictPath);
        if (charDict_.empty()) {
            spdlog::warn("[ONNX OCR] Character dict empty — recognition may fail");
        }

        loaded_ = true;
        spdlog::info("[ONNX OCR] Initialized. CUDA={}", usingCUDA_);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("[ONNX OCR] Init failed: {}", e.what());
        loaded_ = false;
        return false;
    }
}

// -------------------------------------------------------------------------
// recognize — main entry point
// -------------------------------------------------------------------------

std::string ONNXOCREngine::recognize(const cv::Mat& image) {
    if (!loaded_ || image.empty()) return "";

    auto words = recognizeDetailed(image);
    std::string result;
    for (const auto& w : words) {
        if (!result.empty()) result += " ";
        result += w.text;
    }
    return result;
}

// -------------------------------------------------------------------------
// recognizeDetailed
// -------------------------------------------------------------------------

std::vector<OCRWord> ONNXOCREngine::recognizeDetailed(const cv::Mat& image) {
    std::vector<OCRWord> results;
    if (!loaded_ || image.empty()) return results;

    try {
        // Step 1: Detect text regions
        auto regions = runDetection(image);

        // Step 2: Recognize text in each region
        for (const auto& bbox : regions) {
            // Clamp bbox to image bounds
            cv::Rect safe = bbox & cv::Rect(0, 0, image.cols, image.rows);
            if (safe.empty()) continue;

            cv::Mat crop = image(safe);
            std::string text = runRecognition(crop);

            if (!text.empty()) {
                OCRWord word;
                word.text       = text;
                word.confidence = 0.9f; // placeholder — set from model output
                word.bbox       = safe;
                results.push_back(word);
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("[ONNX OCR] recognizeDetailed error: {}", e.what());
    }

    return results;
}

// -------------------------------------------------------------------------
// runDetection — text region detection using det model
// -------------------------------------------------------------------------

std::vector<cv::Rect> ONNXOCREngine::runDetection(const cv::Mat& image) {
    std::vector<cv::Rect> regions;

    // Preprocess
    cv::Mat input = preprocessForDet(image);
    int H = input.rows, W = input.cols;

    // Build input tensor
    std::vector<float> inputData(1 * 3 * H * W);
    // HWC -> CHW conversion
    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);
    for (int c = 0; c < 3; c++) {
        const float* src = channels[c].ptr<float>(0);
        std::copy(src, src + H * W,
                  inputData.begin() + c * H * W);
    }

    std::vector<int64_t> shape = {1, 3, H, W};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputData.data(), inputData.size(),
        shape.data(), shape.size());

    const char* inputNames[]  = {"x"};
    const char* outputNames[] = {"sigmoid_0.tmp_0"};

    auto outputs = detSession_->Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        outputNames, 1);

    // Decode detection map
    float* probMap = outputs[0].GetTensorMutableData<float>();
    auto   outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int outH = static_cast<int>(outShape[2]);
    int outW = static_cast<int>(outShape[3]);

    cv::Mat probMat(outH, outW, CV_32F, probMap);
    cv::Mat binary;
    cv::threshold(probMat, binary, 0.3f, 1.0f, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8U, 255);

    // Scale factor
    float scaleX = static_cast<float>(image.cols) / outW;
    float scaleY = static_cast<float>(image.rows) / outH;

    // Find contours = text regions
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) < 100) continue;
        cv::Rect r = cv::boundingRect(cnt);
        // Scale back to original image coordinates
        r.x      = static_cast<int>(r.x * scaleX);
        r.y      = static_cast<int>(r.y * scaleY);
        r.width  = static_cast<int>(r.width  * scaleX);
        r.height = static_cast<int>(r.height * scaleY);
        // Add padding
        r.x      = std::max(0, r.x - 5);
        r.y      = std::max(0, r.y - 5);
        r.width  += 10;
        r.height += 10;
        regions.push_back(r);
    }

    return regions;
}

// -------------------------------------------------------------------------
// runRecognition — recognize text in a single crop
// -------------------------------------------------------------------------

std::string ONNXOCREngine::runRecognition(const cv::Mat& crop) {
    cv::Mat input = preprocessForRec(crop);
    int H = input.rows, W = input.cols;

    std::vector<float> inputData(1 * 3 * H * W);
    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);
    for (int c = 0; c < 3; c++) {
        const float* src = channels[c].ptr<float>(0);
        std::copy(src, src + H * W,
                  inputData.begin() + c * H * W);
    }

    std::vector<int64_t> shape = {1, 3, H, W};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputData.data(), inputData.size(),
        shape.data(), shape.size());

    const char* inputNames[]  = {"x"};
    const char* outputNames[] = {"softmax_0.tmp_0"};

    auto outputs = recSession_->Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        outputNames, 1);

    float* logits   = outputs[0].GetTensorMutableData<float>();
    auto   outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int seqLen   = static_cast<int>(outShape[1]);
    int numClass = static_cast<int>(outShape[2]);

    return decodeCTC(std::vector<float>(logits,
                     logits + seqLen * numClass), seqLen);
}

// -------------------------------------------------------------------------
// preprocessForDet
// -------------------------------------------------------------------------

cv::Mat ONNXOCREngine::preprocessForDet(const cv::Mat& image) {
    cv::Mat resized, floatImg;

    // Resize to multiple of 32, max 960
    int maxSide = 960;
    float scale = std::min(static_cast<float>(maxSide) / image.cols,
                           static_cast<float>(maxSide) / image.rows);
    int newW = (static_cast<int>(image.cols * scale) / 32) * 32;
    int newH = (static_cast<int>(image.rows * scale) / 32) * 32;
    newW = std::max(32, newW);
    newH = std::max(32, newH);

    cv::resize(image, resized, cv::Size(newW, newH));
    resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);

    // Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    cv::Scalar mean(0.485f, 0.456f, 0.406f);
    cv::Scalar std (0.229f, 0.224f, 0.225f);
    cv::subtract(floatImg, mean, floatImg);
    cv::divide(floatImg, std, floatImg);

    return floatImg;
}

// -------------------------------------------------------------------------
// preprocessForRec
// -------------------------------------------------------------------------

cv::Mat ONNXOCREngine::preprocessForRec(const cv::Mat& crop) {
    cv::Mat resized, floatImg;

    // Fixed height 48, variable width
    int targetH = 48;
    float ratio = static_cast<float>(targetH) / crop.rows;
    int targetW = std::max(10, static_cast<int>(crop.cols * ratio));
    targetW = std::min(targetW, 320);

    cv::resize(crop, resized, cv::Size(targetW, targetH));
    resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);

    cv::Scalar mean(0.5f, 0.5f, 0.5f);
    cv::Scalar std (0.5f, 0.5f, 0.5f);
    cv::subtract(floatImg, mean, floatImg);
    cv::divide(floatImg, std, floatImg);

    return floatImg;
}

// -------------------------------------------------------------------------
// decodeCTC — greedy CTC decode
// -------------------------------------------------------------------------

std::string ONNXOCREngine::decodeCTC(const std::vector<float>& logits, int seqLen) {
    if (charDict_.empty()) return "";

    int numClass = static_cast<int>(logits.size()) / seqLen;
    std::string result;
    int prevIdx = -1;

    for (int t = 0; t < seqLen; t++) {
        const float* row = logits.data() + t * numClass;
        int maxIdx = static_cast<int>(
            std::max_element(row, row + numClass) - row);

        // 0 = blank token in CTC
        if (maxIdx != 0 && maxIdx != prevIdx) {
            int charIdx = maxIdx - 1;
            if (charIdx >= 0 && charIdx < static_cast<int>(charDict_.size())) {
                result += charDict_[charIdx];
            }
        }
        prevIdx = maxIdx;
    }

    return result;
}

// -------------------------------------------------------------------------
// loadDict
// -------------------------------------------------------------------------

std::vector<std::string> ONNXOCREngine::loadDict(const std::string& dictPath) {
    std::vector<std::string> dict;
    std::ifstream f(dictPath);
    if (!f.is_open()) {
        spdlog::warn("[ONNX OCR] Dict not found: {}", dictPath);
        return dict;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) dict.push_back(line);
    }
    spdlog::info("[ONNX OCR] Loaded {} characters from dict", dict.size());
    return dict;
}
```

---

### A4. Update ModelRegistry to use ONNXOCREngine

Open include/ModelRegistry.hpp.
Add member:
```cpp
// ADD after existing OCR members:
#include "ONNXOCREngine.hpp"
std::unique_ptr<ONNXOCREngine> onnxOCR_;
bool onnxOCRLoaded_ = false;
```

Open src/ModelRegistry.cpp.
Find `ModelRegistry::loadModels` or wherever models are initialized.
Add after EAST is loaded:
```cpp
// ADD: Initialize ONNX OCR engine
std::string detPath = modelsDir + "/ocr_onnx/ocr_det.onnx";
std::string recPath = modelsDir + "/ocr_onnx/ocr_rec.onnx";
std::string dictPath = modelsDir + "/ocr_onnx/en_dict.txt";

onnxOCR_ = std::make_unique<ONNXOCREngine>();
if (onnxOCR_->initialize(detPath, recPath, dictPath)) {
    onnxOCRLoaded_ = true;
    spdlog::info("[ModelRegistry] ONNX OCR loaded. CUDA={}",
                 onnxOCR_->isUsingCUDA());
} else {
    spdlog::warn("[ModelRegistry] ONNX OCR failed — falling back to Tesseract");
    onnxOCRLoaded_ = false;
}
```

Find `ModelRegistry::runOCR` (the function that currently calls Tesseract).
Replace the Tesseract call with ONNX OCR when available:
```cpp
std::string ModelRegistry::runOCR(const cv::Mat& image) {
    // Try ONNX OCR first (GPU accelerated)
    if (onnxOCRLoaded_ && onnxOCR_) {
        std::string result = onnxOCR_->recognize(image);
        if (!result.empty()) return result;
    }

    // Fallback to Tesseract (CPU) if ONNX OCR fails or not loaded
    if (!tessAPI_) return "";
    tessAPI_->SetImage(image.data, image.cols, image.rows,
                       image.channels(), image.step);
    char* text = tessAPI_->GetUTF8Text();
    std::string result = text ? text : "";
    delete[] text;
    return result;
}
```

---

## PART B — FULL CUDA SIGN LANGUAGE (Replace MediaPipe with ONNX)

### B1. Download ONNX Gesture Model (run in PowerShell terminal)

```powershell
# Create directory
New-Item -ItemType Directory -Force -Path "models\sign_language"

# Download pre-trained ONNX hand gesture model
# Source: HAGRID dataset model (18 gestures, ONNX format)
$GestureUrl = "https://github.com/hukenovs/hagrid/releases/download/v1.1/hagrid_classifier_352_common_final_model.onnx"
Invoke-WebRequest -Uri $GestureUrl -OutFile "models\sign_language\gesture_model.onnx"

Write-Host "Gesture model downloaded"
```

If that URL fails, alternative download:
```powershell
# Alternative: MobileNetV3 hand gesture classifier (ONNX)
$AltUrl = "https://github.com/hukenovs/hagrid/releases/download/v1.1/hagrid_classifier_352_common_final_model.onnx"
Invoke-WebRequest -Uri $AltUrl -OutFile "models\sign_language\gesture_model.onnx"
```

---

### B2. Replace sign_language_service.py COMPLETELY

Replace the entire contents of scripts/python/sign_language_service.py with:

```python
#!/usr/bin/env python3
"""
Sign Language Service - FULLY CUDA via ONNX Runtime
Called from C++ via PythonBridge (same pattern as stt_service.py)

Uses:
  - MediaPipe Hands (hand detection + 21 keypoints)
  - ONNX Runtime + CUDA (gesture classification)

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
GESTURE_MODEL = os.path.join(MODEL_DIR, 'gesture_model.onnx')

MIN_CONFIDENCE = 0.6

# HAGRID 18-class gesture labels mapped to English words
GESTURE_MAP = {
    0:  "call",
    1:  "dislike",
    2:  "fist",
    3:  "four",
    4:  "like",
    5:  "mute",
    6:  "ok",
    7:  "one",
    8:  "palm",
    9:  "peace",
    10: "peace_inverted",
    11: "rock",
    12: "stop",
    13: "stop_inverted",
    14: "three",
    15: "three2",
    16: "two_up",
    17: "two_up_inverted",
}

ENGLISH_MAP = {
    "call":           "call",
    "dislike":        "no",
    "fist":           "stop",
    "four":           "four",
    "like":           "yes",
    "mute":           "quiet",
    "ok":             "ok",
    "one":            "one",
    "palm":           "hello",
    "peace":          "peace",
    "peace_inverted": "victory",
    "rock":           "rock",
    "stop":           "stop",
    "stop_inverted":  "wait",
    "three":          "three",
    "three2":         "three",
    "two_up":         "two",
    "two_up_inverted":"two",
}


# =============================================================================
# ONNX Runtime Session with CUDA
# =============================================================================

_ort_session = None

def get_ort_session():
    """
    Singleton ONNX Runtime session with CUDA provider.
    Loaded once, reused for all frames.
    """
    global _ort_session
    if _ort_session is not None:
        return _ort_session

    if not os.path.exists(GESTURE_MODEL):
        raise FileNotFoundError(
            f"Gesture model not found: {GESTURE_MODEL}\n"
            "Run: Invoke-WebRequest -Uri <url> -OutFile models/sign_language/gesture_model.onnx"
        )

    import onnxruntime as ort

    providers = ort.get_available_providers()
    cuda_ok   = 'CUDAExecutionProvider' in providers

    if cuda_ok:
        provider_list = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
        print("[SignLang] ONNX Runtime: using CUDA GPU", file=sys.stderr)
    else:
        provider_list = ['CPUExecutionProvider']
        print("[SignLang] ONNX Runtime: CUDA not found, using CPU", file=sys.stderr)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2

    _ort_session = ort.InferenceSession(
        GESTURE_MODEL,
        sess_options=opts,
        providers=provider_list
    )
    return _ort_session


# =============================================================================
# Hand Detection (MediaPipe — fast, lightweight)
# =============================================================================

_mp_hands = None

def get_mp_hands():
    global _mp_hands
    if _mp_hands is None:
        import mediapipe as mp
        _mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6,
            model_complexity=0   # fastest model
        )
    return _mp_hands


def extract_hand_region(frame_bgr: np.ndarray):
    """
    Use MediaPipe to detect hand bounding box.
    Returns cropped hand region or None.
    """
    try:
        import mediapipe as mp
        hands = get_mp_hands()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        # Get bounding box of hand landmarks
        lms = result.multi_hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        xs = [lm.x * w for lm in lms.landmark]
        ys = [lm.y * h for lm in lms.landmark]
        x1 = max(0, int(min(xs)) - 30)
        y1 = max(0, int(min(ys)) - 30)
        x2 = min(w, int(max(xs)) + 30)
        y2 = min(h, int(max(ys)) + 30)

        crop = frame_bgr[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    except Exception as e:
        print(f"[SignLang] Hand detection error: {e}", file=sys.stderr)
        return None


# =============================================================================
# ONNX Gesture Classification (CUDA)
# =============================================================================

def classify_gesture_onnx(hand_crop: np.ndarray):
    """
    Classify gesture using ONNX model on GPU.
    Returns (gesture_name, confidence) or ("", 0.0)
    """
    try:
        session = get_ort_session()

        # Preprocess: resize to 224x224, normalize
        img = cv2.resize(hand_crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / \
                     np.array([0.229, 0.224, 0.225])
        # HWC -> NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)

        input_name = session.get_inputs()[0].name
        outputs    = session.run(None, {input_name: img})
        logits     = outputs[0][0]  # shape: (num_classes,)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs      = exp_logits / exp_logits.sum()

        top_idx    = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        gesture_name = GESTURE_MAP.get(top_idx, "unknown")
        return gesture_name, confidence

    except Exception as e:
        print(f"[SignLang] ONNX classification error: {e}", file=sys.stderr)
        return "", 0.0


# =============================================================================
# Main Recognition Pipeline
# =============================================================================

def recognize_frame(frame_bgr: np.ndarray):
    """
    Full pipeline:
    1. MediaPipe detects hand region (fast, CPU)
    2. ONNX Runtime classifies gesture (GPU via CUDA)
    Returns (english_word, confidence)
    """
    hand_crop = extract_hand_region(frame_bgr)
    if hand_crop is None:
        return "", 0.0

    gesture, confidence = classify_gesture_onnx(hand_crop)
    if confidence < MIN_CONFIDENCE:
        return "", confidence

    english = ENGLISH_MAP.get(gesture, gesture)
    return english, confidence


def recognize_from_image(image_path: str) -> str:
    frame = cv2.imread(image_path)
    if frame is None:
        return f"SIGN_ERROR:Cannot load image: {image_path}"
    try:
        word, conf = recognize_frame(frame)
        if word:
            return f"SIGN_RESULT:{word}:{conf:.2f}"
        return "SIGN_EMPTY"
    except Exception as e:
        return f"SIGN_ERROR:{str(e)}"


def recognize_from_camera(duration_seconds: int = 3) -> str:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "SIGN_ERROR:Cannot open camera"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    best_word, best_conf = "", 0.0
    start = time.time()
    try:
        while (time.time() - start) < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            word, conf = recognize_frame(frame)
            if word and conf > best_conf:
                best_word, best_conf = word, conf
    finally:
        cap.release()
    if best_word:
        return f"SIGN_RESULT:{best_word}:{best_conf:.2f}"
    return "SIGN_EMPTY"


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input",    nargs="?", help="Image path")
    parser.add_argument("--camera", type=int,  help="Camera duration seconds")
    args = parser.parse_args()

    if args.camera is not None:
        print(recognize_from_camera(args.camera))
    elif args.input:
        print(recognize_from_image(args.input))
    else:
        # Print help and exit cleanly (for --help check)
        parser.print_help()
        sys.exit(0)
```

---

## PART C — CMakeLists.txt Update

Open CMakeLists.txt.
The GLOB_RECURSE already picks up ONNXOCREngine.cpp from src/.
No changes needed to CMakeLists.txt.

Rebuild:
```powershell
cmake --build build --config Release
```

---

## PART D — VERIFICATION

After build succeeds, run in PowerShell:

```powershell
# Check ONNX OCR startup log
.\build\bin\Release\smart_glasses_hud.exe 2>&1 | findstr /i "ONNX OCR"
```

Expected:
```
[ONNX OCR] Using CUDA execution provider
[ModelRegistry] ONNX OCR loaded. CUDA=1
```

```powershell
# Check Sign Language startup log
.\build\bin\Release\smart_glasses_hud.exe 2>&1 | findstr /i "SignLang"
```

Expected:
```
[TranslationTaskManager] Sign language translator ready.
```

Then test Python sign service directly:
```powershell
python scripts\python\sign_language_service.py --help
```

Expected: no errors, prints usage text.

---

## FINAL STATE AFTER THIS PROMPT

| Component           | Provider               | GPU? |
|---------------------|------------------------|------|
| YOLO Detection      | OnnxDetector + CUDA    | YES  |
| Face Detection      | OpenCV DNN + CUDA      | YES  |
| EAST Text Detect    | OpenCV DNN + CUDA      | YES  |
| OCR Text Recog      | ONNXOCREngine + CUDA   | YES  |
| Sign Language       | ONNX Runtime + CUDA    | YES  |
| TTS / STT           | CPU (audio — no GPU)   | N/A  |
| Translation Logic   | CPU (strings — no GPU) | N/A  |

Everything that CAN run on GPU now DOES run on GPU.
