#pragma once

#include <string>
#include <vector>

namespace SmartGlasses {

// Thin helper for calling the Python helper scripts in scripts/python/.
// All functions return an empty string on failure; callers can then fall
// back to existing behaviour or show an error in the HUD.
class PythonBridge {
public:
    // Set base directory where scripts/python lives (usually exeDir or source dir).
    // If not set, functions will assume the current working directory.
    static void setBaseDir(const std::string& dir);

    // Text-to-speech: fire-and-forget, no return payload.
    static void speakText(const std::string& text);

    // STT: record for up to durationSec and return recognised text.
    static std::string runSTT(int durationSec);
    // STT from WAV file (push-to-talk: record in C++, then transcribe file).
    static std::string runSTTFromFile(const std::string& wavPath);

    // OCR: write frame image to a temporary file and return recognised text.
    static std::string runOCR(const std::string& imagePath);

    // GPU-accelerated OCR via ocr_onnx_service.py (RapidOCR + ONNX Runtime CUDA).
    static std::string runOCRonnx(const std::string& imagePath);

    // GPU-accelerated sign language via sign_onnx_service.py.
    static std::string runSignOnnx(const std::string& imagePath);

private:
    static std::string getScriptsDir();
    static int runPython(const std::vector<std::string>& args, std::string& stdoutOut, std::string& stderrOut);
};

} // namespace SmartGlasses

