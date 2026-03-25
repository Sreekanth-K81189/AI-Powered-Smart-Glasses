#pragma once
// SpeechService.hpp - Windows SAPI Speech-to-Text
// Uses SpInprocRecognizer - no external library needed.
#ifndef SPEECH_SERVICE_HPP
#define SPEECH_SERVICE_HPP
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
namespace SmartGlasses {
class SpeechService {
public:
    using ResultCallback = std::function<void(const std::string&)>;
    SpeechService();
    ~SpeechService();
    bool startListening(ResultCallback cb);
    void stopListening();
    bool isListening() const { return listening_; }
    std::string getLastResult();
private:
    void listenLoop();
    std::thread       thread_;
    std::atomic<bool> listening_{false};
    ResultCallback    callback_;
    std::string       lastResult_;
    mutable std::mutex mutex_;
};
} // namespace SmartGlasses
#endif