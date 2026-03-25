#pragma once
#include <thread>
#include <atomic>
#include <string>
#include <chrono>

class TranslationEngine {
public:
    TranslationEngine();
    ~TranslationEngine();

    // Start the drain thread — call once from Pipeline::start()
    void start();

    // Stop the drain thread — call from Pipeline::stop()
    // Signals gTranslationQueue to unblock, then joins the thread
    void stop();

    // Returns true if the translation HTTP endpoint is reachable
    // (ping test on startup — result logged to camera_log.txt)
    bool isAvailable() const;

private:
    void run();   // thread entry point — drains gTranslationQueue forever

    // Calls LibreTranslate POST /translate
    // Returns translated string or "" on failure
    std::string translate(const std::string& text,
                          const std::string& sourceLang,
                          const std::string& targetLang);

    // Deduplication check
    // Returns true if this text was already spoken within TRANSLATE_DEDUP_MS
    bool isDuplicate(const std::string& text) const;

    std::thread             thread_;
    std::atomic<bool>       running_    {false};
    std::atomic<bool>       available_  {false};

    // Dedup state
    std::string             lastSpoken_;
    std::chrono::steady_clock::time_point lastSpokenTime_;
};

