#pragma once
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <vector>

namespace hud {

using SttCallback = std::function<void(const std::string& text)>;

class SttEngine {
public:
    explicit SttEngine(const std::string& whisperModelPath);
    ~SttEngine();

    void Start(SttCallback cb);
    void Stop();
    bool IsListening() const { return listening_.load(); }

    // One-shot transcription from file
    std::string TranscribeFile(const std::string& wavPath);

private:
    std::string modelPath_;
    SttCallback callback_;
    std::thread worker_;
    std::atomic<bool> listening_{false};
    std::atomic<bool> running_{true};

    void* whisperCtx_ = nullptr;
    void WorkerLoop();
    std::vector<float> CaptureAudio(int durationMs = 3000);
    std::string RunWhisper(const std::vector<float>& audio);
};

} // namespace hud
