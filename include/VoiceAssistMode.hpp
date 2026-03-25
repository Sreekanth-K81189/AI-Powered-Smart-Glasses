#pragma once
/*
 * VoiceAssistMode.hpp
 * Dedicated mode for visually impaired users.
 * Continuously runs scene analysis and speaks descriptions at a
 * configured interval. Completely independent of the HUD task panel.
 */

#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <opencv2/core.hpp>

namespace SmartGlasses {

class Pipeline;
class TTSService;
class CameraManager;

class VoiceAssistMode {
public:
    void setVoiceEnabled(bool v) { voiceEnabled_.store(v); }

    VoiceAssistMode(CameraManager& camera, Pipeline& pipeline, TTSService& tts);
    ~VoiceAssistMode();

    // Start the background voice-assist thread
    void start();

    // Stop and join the thread
    void stop();

    bool isRunning() const { return running_; }

    // Set how many seconds between each full scene description (default 4s)
    void setSpeakInterval(double seconds) { speakIntervalSec_ = seconds; }

    // Latest status string for HUD display
    std::string getLastDescription() const;

private:
    std::string lastVADescription_;
    double      vaForceTimer_ = 0.0;
    std::atomic<bool> voiceEnabled_{true};

    void loop_();

    CameraManager* camera_   = nullptr;
    Pipeline*      pipeline_ = nullptr;
    TTSService*    tts_      = nullptr;

    std::atomic<bool>  running_{false};
    std::thread        thread_;
    double             speakIntervalSec_ = 4.0;

    mutable std::mutex descMutex_;
    std::string        lastDescription_;
};

} // namespace SmartGlasses
