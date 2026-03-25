#pragma once
#include <thread>
#include <atomic>
#include <string>
#include <vector>

class TTSEngine {
public:
    TTSEngine();
    ~TTSEngine();

    // Locate Piper exe, start drain thread.
    // Returns true always — stub mode if Piper not found.
    bool start();

    // Stop drain thread and any active PortAudio playback.
    void stop();

    bool isRunning()  const { return running_.load(); }
    bool hasPiper()   const { return hasPiper_; }

    std::string statusString() const;

private:
    void run();

    std::vector<float> synthesise(const std::string& text);
    void playAudio(std::vector<float>& samples, int sampleRate);

    std::thread       thread_;
    std::atomic<bool> running_  {false};
    bool              hasPiper_ = false;
};

extern TTSEngine gTTSEngine;

