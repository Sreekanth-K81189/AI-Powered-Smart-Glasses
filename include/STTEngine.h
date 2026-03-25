#pragma once
#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include <mutex>

// Forward declare whisper context to avoid including whisper.h in every TU
struct whisper_context;

class STTEngine {
public:
    enum class Engine { WHISPER, SAPI, NONE };

    STTEngine();
    ~STTEngine();

    // Load Whisper model and start transcription thread.
    // Returns true on success, false if running only in stub mode.
    bool start();

    // Stop transcription thread and free Whisper context.
    void stop();

    bool isRunning() const { return running_.load(); }

    // True if a valid Whisper model is loaded.
    bool hasModel() const { return hasWhisper_ && ctx_ != nullptr; }

    Engine getEngine() const { return activeEngine_; }

    // Returns a human-readable status string for logging.
    std::string statusString() const;

    // ----- On-demand listen -----
    // pushToTalk: true = record while Space held (AudioCapture), release = stop and transcribe.
    // pushToTalk: false = run Python STT for up to 30s (cannot interrupt).
    void startListening(bool pushToTalk = false);
    void stopListening();
    bool isListening() const { return isRecording_.load(); }

private:
    void run();
    void recordLoop();
    void recordLoopPushToTalk();

    // Returns true if RMS energy of samples exceeds VAD threshold
    bool detectVoiceActivity(const std::vector<float>& samples) const;

    // Runs whisper_full on samples, returns transcript string.
    // Returns "" on failure or in stub mode.
    std::string transcribe(const std::vector<float>& samples);

    // Unicode-range language detection — no external library required.
    std::string detectLanguage(const std::string& text) const;

    whisper_context*    ctx_        = nullptr;
    std::thread         thread_;
    std::atomic<bool>   running_   {false};
    bool                hasWhisper_ = false;
    Engine              activeEngine_ = Engine::NONE;

    std::atomic<bool>   isRecording_{false};
    std::atomic<bool>   pushToTalkMode_{false};
    std::thread         recordThread_;
    std::vector<float>  recordBuffer_;
    std::mutex          recordMutex_;
};

extern STTEngine gSTTEngine;

namespace SmartGlasses {
// SmartGlasses::asyncSTT — blocking STT for durationSec seconds via Python bridge; used by HUD async wrapper.
std::string asyncSTT(int durationSec);
}

