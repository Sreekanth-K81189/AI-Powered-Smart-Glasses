#pragma once
// =============================================================================
// SignLanguageTranslator.hpp
// include/SignLanguageTranslator.hpp
//
// Real-time ASL sign language recognition via Python bridge.
// Follows the same pattern as PythonBridge (STT/TTS/OCR services).
//
// Usage in main.cpp / TranslationTaskManager.cpp:
//     SignLanguageTranslator slt;
//     slt.setBaseDir(baseDir);
//
//     // From camera frame (real-time):
//     std::string word = slt.recognizeFromFrame(frame);
//
//     // From image file:
//     std::string word = slt.recognizeFromImage(imagePath);
// =============================================================================

#include <string>
#include <atomic>
#include <mutex>
#include <future>
#include <opencv2/core.hpp>

namespace SmartGlasses {

struct SignRecognitionResult {
    std::string word;        // Recognized English word (empty if none)
    float       confidence;  // 0.0 - 1.0
    bool        detected;    // true if a hand was found in frame
};

class SignLanguageTranslator {
public:

    // -------------------------------------------------------------------------
    // Constructor / Destructor
    // -------------------------------------------------------------------------
    SignLanguageTranslator();
    ~SignLanguageTranslator() = default;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// Set project base directory (same as PythonBridge::setBaseDir)
    void setBaseDir(const std::string& baseDir);

    /// Minimum confidence threshold for accepting a prediction (default 0.6)
    void setConfidenceThreshold(float threshold);

    // -------------------------------------------------------------------------
    // Recognition
    // -------------------------------------------------------------------------

    /// Recognize sign from an OpenCV frame (saves temp file, calls Python service)
    /// This is the main method to call each frame in real-time mode.
    SignRecognitionResult recognizeFromFrame(const cv::Mat& frame);

    /// Recognize sign from an existing image file path
    SignRecognitionResult recognizeFromImage(const std::string& imagePath);

    /// Recognize using camera capture for N seconds (blocking)
    SignRecognitionResult recognizeFromCamera(int durationSeconds = 3);

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    /// Returns true if Python service and models are available
    bool isAvailable() const;

    /// Last recognized word (cached)
    std::string getLastWord() const { return lastResult_.word; }

    /// Last confidence score
    float getLastConfidence() const { return lastResult_.confidence; }

    // Non-blocking: starts recognition in background, returns last cached result
    SignRecognitionResult recognizeAsync(const cv::Mat& frame);

private:
    std::string baseDir_;
    float confidenceThreshold_ = 0.6f;
    SignRecognitionResult lastResult_;
    bool available_ = false;

    // Async recognition state
    std::future<SignRecognitionResult> asyncFuture_;
    std::atomic<bool>                 asyncRunning_{ false };
    cv::Mat                           asyncFrame_;
    mutable std::mutex                asyncMutex_;

    /// Run Python sign language service and return raw output
    std::string runPythonService(const std::string& args) const;

    /// Parse "SIGN_RESULT:<word>:<confidence>" output from Python service
    SignRecognitionResult parseServiceOutput(const std::string& raw) const;

    /// Execute shell command and capture stdout (reuses PythonBridge pattern)
    std::string runCmd(const std::string& cmd) const;

    /// Get path to sign_language_service.py
    std::string getServicePath() const;

    /// Get temp frame path for inter-process frame passing
    std::string getTempFramePath() const;
};

} // namespace SmartGlasses

