#pragma once
#include <string>
#include <atomic>
#include <thread>
#include <functional>

namespace hud {

using SceneDescriber = std::function<std::string()>;

class VoiceAssistMode {
public:
    VoiceAssistMode();
    ~VoiceAssistMode();

    void Start(SceneDescriber describer, int intervalSeconds = 4);
    void Stop();
    void Toggle(SceneDescriber describer, int intervalSeconds = 4);
    bool IsRunning() const { return running_.load(); }
    void SetInterval(int seconds) { interval_ = seconds; }

private:
    std::atomic<bool> running_{false};
    std::thread       worker_;
    int               interval_ = 4;
    void WorkerLoop(SceneDescriber describer);
};

} // namespace hud
