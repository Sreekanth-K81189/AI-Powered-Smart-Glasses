// =============================================================================
// SignLanguageTranslator.cpp
// src/SignLanguageTranslator.cpp
//
// Real-time ASL word recognition via Python bridge.
// Follows the exact same pattern as PythonBridge.cpp (STT/TTS/OCR).
//
// Integration steps for Cursor:
//   1. Copy SignLanguageTranslator.hpp -> include/SignLanguageTranslator.hpp
//   2. Copy SignLanguageTranslator.cpp -> src/SignLanguageTranslator.cpp
//   3. Copy sign_language_service.py   -> scripts/python/sign_language_service.py
//   4. In CMakeLists.txt add:
//        src/SignLanguageTranslator.cpp
//      to the target_sources list (same place as PythonBridge.cpp)
//   5. In TranslationTaskManager.cpp or main.cpp:
//        #include "SignLanguageTranslator.hpp"
//        SignLanguageTranslator signLT;
//        signLT.setBaseDir(s_baseDir);
//        auto result = signLT.recognizeFromFrame(frame);
//        if (result.detected) gResultsStore.setSignWord(result.word);
//   6. In your HUD overlay code, display gResultsStore.getSignWord()
// =============================================================================

#include "SignLanguageTranslator.hpp"

#include <spdlog/spdlog.h>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <future>
#include <chrono>

#ifdef _WIN32
    #include <windows.h>
    #define POPEN  _popen
    #define PCLOSE _pclose
    #define PATH_SEP "\\"
#else
    #include <unistd.h>
    #define POPEN  popen
    #define PCLOSE pclose
    #define PATH_SEP "/"
#endif

namespace SmartGlasses {

namespace fs = std::filesystem;

// =============================================================================
// Constructor
// =============================================================================

SignLanguageTranslator::SignLanguageTranslator() {
    // Check if Python is available
    std::string check = runCmd("python3 --version 2>&1");
    if (check.find("Python") != std::string::npos) {
        available_ = true;
        spdlog::info("[SignLang] Python found: {}", check);
    } else {
        available_ = false;
        spdlog::warn("[SignLang] Python3 not found. Sign language translation disabled.");
    }
}

// =============================================================================
// Configuration
// =============================================================================

void SignLanguageTranslator::setBaseDir(const std::string& baseDir) {
    baseDir_ = baseDir;
    spdlog::info("[SignLang] Base directory set: {}", baseDir_);

    // Verify model files exist
    std::string modelDir = baseDir_ + PATH_SEP "models" PATH_SEP "sign_language";
    bool handModel    = fs::exists(modelDir + PATH_SEP "hand_landmarker.task");
    bool gestureModel = fs::exists(modelDir + PATH_SEP "gesture_recognizer.task");

    if (!handModel || !gestureModel) {
        spdlog::warn("[SignLang] Models missing in {}. Run install_sign_language.sh", modelDir);
        available_ = false;
    } else {
        spdlog::info("[SignLang] Models found. Ready.");
        available_ = true;
    }
}

void SignLanguageTranslator::setConfidenceThreshold(float threshold) {
    confidenceThreshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

// =============================================================================
// recognizeFromFrame
// Primary real-time method. Called each frame from TranslationTaskManager.
//
// Flow:
//   1. Save OpenCV frame to temp .jpg (same as OCR in TranslationTaskManager.cpp)
//   2. Call Python service with temp file path
//   3. Parse output
//   4. Delete temp file
//   5. Return result
// =============================================================================

SignRecognitionResult SignLanguageTranslator::recognizeFromFrame(const cv::Mat& frame) {
    SignRecognitionResult empty;
    empty.detected   = false;
    empty.confidence = 0.0f;
    empty.word       = "";

    if (!available_ || frame.empty()) {
        return empty;
    }

    // Save frame to temp file
    std::string tempPath = getTempFramePath();

    bool saved = cv::imwrite(tempPath, frame);
    if (!saved) {
        spdlog::error("[SignLang] Failed to write temp frame: {}", tempPath);
        return empty;
    }

    // Call Python service
    std::string raw = runPythonService("\"" + tempPath + "\"");

    // Delete temp file
    std::remove(tempPath.c_str());

    // Parse and return
    SignRecognitionResult result = parseServiceOutput(raw);

    // Cache result
    if (result.detected && result.confidence >= confidenceThreshold_) {
        lastResult_ = result;
        spdlog::debug("[SignLang] Recognized: '{}' ({:.0f}%)",
                      result.word, result.confidence * 100.0f);
    }

    return result;
}

SignRecognitionResult SignLanguageTranslator::recognizeAsync(const cv::Mat& frame) {
    // Always return the last known result immediately (non-blocking)
    SignRecognitionResult current = lastResult_;

    // If a recognition is already running in background, check if it's done
    if (asyncRunning_.load()) {
        if (asyncFuture_.valid() &&
            asyncFuture_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            try {
                SignRecognitionResult newResult = asyncFuture_.get();
                asyncRunning_.store(false);
                if (newResult.detected && newResult.confidence >= confidenceThreshold_) {
                    lastResult_ = newResult;
                    current     = newResult;
                }
            } catch (...) {
                asyncRunning_.store(false);
            }
        }
        // Either still running or just finished — return cached result
        return current;
    }

    // No recognition running — start a new one in the background
    if (frame.empty() || !available_) return current;

    asyncRunning_.store(true);

    // Capture a downscaled copy of the frame on the background thread to avoid extra copies here.
    asyncFuture_ = std::async(std::launch::async,
        [this, frame]() -> SignRecognitionResult {
            return this->recognizeFromFrame(frame);
        });

    return current;
}

// =============================================================================
// recognizeFromImage
// =============================================================================

SignRecognitionResult SignLanguageTranslator::recognizeFromImage(const std::string& imagePath) {
    SignRecognitionResult empty;
    empty.detected   = false;
    empty.confidence = 0.0f;
    empty.word       = "";

    if (!available_) return empty;

    std::string raw = runPythonService("\"" + imagePath + "\"");
    return parseServiceOutput(raw);
}

// =============================================================================
// recognizeFromCamera
// Blocking mode: captures from camera for N seconds, returns best result
// =============================================================================

SignRecognitionResult SignLanguageTranslator::recognizeFromCamera(int durationSeconds) {
    SignRecognitionResult empty;
    empty.detected   = false;
    empty.confidence = 0.0f;
    empty.word       = "";

    if (!available_) return empty;

    std::string raw = runPythonService("--camera " + std::to_string(durationSeconds));
    return parseServiceOutput(raw);
}

// =============================================================================
// isAvailable
// =============================================================================

bool SignLanguageTranslator::isAvailable() const {
    return available_;
}

// =============================================================================
// Private: runPythonService
// =============================================================================

std::string SignLanguageTranslator::runPythonService(const std::string& args) const {
    // Use GPU-accelerated sign_onnx_service.py (via MediaPipe + ONNX Runtime GPU)
    std::string scriptPath;
    if (baseDir_.empty()) {
        scriptPath = std::string("scripts") + PATH_SEP + "python" + PATH_SEP + "sign_onnx_service.py";
    } else {
        scriptPath = baseDir_ + PATH_SEP "scripts" PATH_SEP "python" PATH_SEP "sign_onnx_service.py";
    }

    std::string cmd;

#ifdef _WIN32
    cmd = "py \"" + scriptPath + "\" " + args + " 2>nul";
#else
    cmd = "python3 \"" + scriptPath + "\" " + args + " 2>/dev/null";
#endif

    spdlog::debug("[SignLang] Running: {}", cmd);
    return runCmd(cmd);
}

// =============================================================================
// Private: parseServiceOutput
// Parses output from sign_language_service.py:
//   "SIGN_RESULT:<word>:<confidence>"  -> populated result
//   "SIGN_EMPTY"                       -> not detected
//   "SIGN_ERROR:<message>"             -> error
// =============================================================================

SignRecognitionResult SignLanguageTranslator::parseServiceOutput(const std::string& raw) const {
    SignRecognitionResult result;
    result.detected   = false;
    result.confidence = 0.0f;
    result.word       = "";

    if (raw.empty()) {
        return result;
    }

    // Trim whitespace
    std::string trimmed = raw;
    trimmed.erase(trimmed.find_last_not_of(" \n\r\t") + 1);
    trimmed.erase(0, trimmed.find_first_not_of(" \n\r\t"));

    if (trimmed.rfind("SIGN_RESULT:", 0) == 0) {
        // Format: SIGN_RESULT:<word>:<confidence>
        std::string payload = trimmed.substr(12); // after "SIGN_RESULT:"
        size_t sep = payload.rfind(':');           // last colon separates word from confidence

        if (sep != std::string::npos) {
            result.word       = payload.substr(0, sep);
            result.confidence = std::stof(payload.substr(sep + 1));
            result.detected   = true;

            // Replace underscores with spaces for display
            std::replace(result.word.begin(), result.word.end(), '_', ' ');
        }

    } else if (trimmed == "SIGN_EMPTY") {
        result.detected = false;

    } else if (trimmed.rfind("SIGN_ERROR:", 0) == 0) {
        spdlog::warn("[SignLang] Service error: {}", trimmed.substr(11));
    } else {
        // Unexpected output — log it
        spdlog::debug("[SignLang] Unexpected output: {}", trimmed);
    }

    return result;
}

// =============================================================================
// Private: runCmd
// Execute shell command and capture stdout.
// Identical to PythonBridge::runCmd pattern.
// =============================================================================

std::string SignLanguageTranslator::runCmd(const std::string& cmd) const {
    FILE* pipe = POPEN(cmd.c_str(), "r");
    if (!pipe) return "";

    char   buffer[256];
    std::string result;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

    PCLOSE(pipe);

    // Trim trailing whitespace
    result.erase(result.find_last_not_of(" \n\r\t") + 1);
    return result;
}

// =============================================================================
// Private: getServicePath
// =============================================================================

std::string SignLanguageTranslator::getServicePath() const {
    if (baseDir_.empty()) {
        return "scripts" PATH_SEP "python" PATH_SEP "sign_language_service.py";
    }
    return baseDir_ + PATH_SEP "scripts" PATH_SEP "python" PATH_SEP "sign_language_service.py";
}

// =============================================================================
// Private: getTempFramePath
// =============================================================================

std::string SignLanguageTranslator::getTempFramePath() const {
#ifdef _WIN32
    return std::string(std::getenv("TEMP") ? std::getenv("TEMP") : "C:\\Temp")
           + "\\sign_frame_temp.jpg";
#else
    return "/tmp/sign_frame_temp.jpg";
#endif
}

} // namespace SmartGlasses

