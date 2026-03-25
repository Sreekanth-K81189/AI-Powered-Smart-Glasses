#pragma once

#include <string>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

#include "Config.hpp"

namespace SmartGlasses {

enum class TTSBackend { TTS_WINRT, TTS_POWERSHELL, TTS_SAPI, TTS_NONE };

struct LogEntry { std::string text; std::string timestamp; };

class TTSService {
public:
    TTSService();
    ~TTSService();

    void speak(const std::string& text, bool forceUserAction = false);  // non-blocking; forceUserAction=true bypasses TTS_ENABLED
    void stop();
    bool isEnabled() const { return Config::TTS_ENABLED; }
    bool isAvailable() const { return activeBackend_ != TTSBackend::TTS_NONE; }
    std::deque<LogEntry> getLog() const;
    TTSBackend backend() const { return activeBackend_; }

private:
    std::queue<std::string> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> running_{true};
    std::deque<LogEntry> log_;
    TTSBackend activeBackend_ = TTSBackend::TTS_NONE;

    void workerLoop();
    void logBackendToFile(const char* backendName);
};

} // namespace SmartGlasses
