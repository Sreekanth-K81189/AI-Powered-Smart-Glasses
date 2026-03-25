#pragma once
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace hud {

class TtsEngine {
public:
    TtsEngine();
    ~TtsEngine();

    void Speak(const std::string& text, int priority = 0);
    void SpeakImmediate(const std::string& text); // interrupt queue
    void Stop();
    void SetRate(int rate);   // words per minute
    void SetVolume(float v);  // 0.0 - 1.0
    bool IsSpeaking() const { return speaking_.load(); }

    // Throttle: don't repeat same message within N seconds
    void SetThrottle(float seconds) { throttleSec_ = seconds; }

private:
    struct Item { std::string text; int priority; };
    std::queue<Item>         queue_;
    std::mutex               mtx_;
    std::condition_variable  cv_;
    std::thread              worker_;
    std::atomic<bool>        running_{true};
    std::atomic<bool>        speaking_{false};
    float                    throttleSec_ = 2.0f;
    std::string              lastSpoken_;
    float                    lastSpokenTime_ = 0.0f;
    int                      rate_   = 175;
    float                    volume_ = 1.0f;
    bool                     useESpeak_ = false;
    std::string              eSpeakPath_;

    void WorkerLoop();
    void SpeakSAPI(const std::string& text);
    void SpeakESpeak(const std::string& text);
};

} // namespace hud
