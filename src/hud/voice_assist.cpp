#include "hud/voice_assist.h"
#include <chrono>
#include <thread>

namespace hud {

VoiceAssistMode::VoiceAssistMode() {}
VoiceAssistMode::~VoiceAssistMode() { Stop(); }

void VoiceAssistMode::Start(SceneDescriber describer, int intervalSeconds) {
    if (running_) return;
    interval_ = intervalSeconds;
    running_  = true;
    worker_   = std::thread(&VoiceAssistMode::WorkerLoop, this, describer);
}

void VoiceAssistMode::Stop() {
    running_ = false;
    if (worker_.joinable()) worker_.join();
}

void VoiceAssistMode::Toggle(SceneDescriber describer, int intervalSeconds) {
    if (running_) Stop(); else Start(describer, intervalSeconds);
}

void VoiceAssistMode::WorkerLoop(SceneDescriber describer) {
    while (running_) {
        std::string desc = describer();
        // TTS is called externally via callback; desc is passed to caller
        (void)desc;
        std::this_thread::sleep_for(std::chrono::seconds(interval_));
    }
}

} // namespace hud
