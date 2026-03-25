#pragma once
#include <string>

namespace SmartGlasses {

class PythonBridge {
public:
    static void setBaseDir(const std::string& dir);
    static void speakText(const std::string& text);
    static std::string runSTT(int durationSec = 7);
    static std::string runSTTFromFile(const std::string& wavPath);
    static std::string runOCR(const std::string& imagePath);

    // GPU-accelerated OCR via ocr_onnx_service.py (RapidOCR + ONNX Runtime CUDA).
    static std::string runOCRonnx(const std::string& imagePath);

    // GPU-accelerated sign language via sign_onnx_service.py.
    static std::string runSignOnnx(const std::string& imagePath);
};

} // namespace SmartGlasses
