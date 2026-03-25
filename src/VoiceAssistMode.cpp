/*
 * VoiceAssistMode.cpp
 * Dedicated voice-only assistance mode for blind / visually impaired users.
 *
 * Runs in a background thread and periodically:
 *  - grabs a frame
 *  - runs obstacle + face detection
 *  - builds a short spoken description
 *  - sends it to TTS (throttled to avoid repeating trivial phrases)
 */
 
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#endif
 
#include <chrono>
#include <iostream>
#include <sstream>

#include <opencv2/core.hpp>

#include "CameraManager.hpp"
#include "Pipeline.hpp"
#include "TTSService.hpp"
#include "VoiceAssistMode.hpp"
 
 namespace SmartGlasses {
 
 VoiceAssistMode::VoiceAssistMode(CameraManager& camera,
                                  Pipeline&       pipeline,
                                  TTSService&     tts)
     : camera_(&camera), pipeline_(&pipeline), tts_(&tts) {}
 
 VoiceAssistMode::~VoiceAssistMode() { stop(); }
 
 void VoiceAssistMode::start() {
     if (running_) return;
     running_ = true;
     thread_  = std::thread(&VoiceAssistMode::loop_, this);
     std::cout << "[VoiceAssist] Started (interval=" << speakIntervalSec_ << "s)\n";
     if (tts_) tts_->speak("Voice assist mode activated");
 }
 
 void VoiceAssistMode::stop() {
     if (!running_) return;
     running_ = false;
     if (thread_.joinable()) thread_.join();
     std::cout << "[VoiceAssist] Stopped\n";
 }
 
 std::string VoiceAssistMode::getLastDescription() const {
     std::lock_guard<std::mutex> lk(descMutex_);
     return lastDescription_;
 }
 
void VoiceAssistMode::loop_() {
    using Clock = std::chrono::steady_clock;
    Clock::time_point nextSpeak = Clock::now();

    while (running_) {
        auto now = Clock::now();

        if (now >= nextSpeak) {
            nextSpeak = now + std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::duration<double>(speakIntervalSec_));

            // Grab frame from camera (non-blocking). If we cannot get a frame,
            // just wait a bit and try again on the next loop.
            if (!camera_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            cv::Mat frame;
            camera_->readFrame(frame);
            if (frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            // Obstacle detection
            cv::Mat obstacleCopy = frame.clone();
            MoveSafeResult moveResult = pipeline_->runObstacleDetection(obstacleCopy, false);

            // Face detection
            cv::Mat faceCopy = frame.clone();
            std::vector<cv::Rect> faces;
            pipeline_->runFaceRecognition(faceCopy, faces, false);

            // Build spoken description
            std::ostringstream desc;
            desc << moveResult.hint;

            if (!faces.empty()) {
                desc << ". ";
                if (faces.size() == 1) desc << "One person detected";
                else desc << faces.size() << " people detected";
            }

            if (moveResult.leftCount > 0 && moveResult.centerCount == 0 && moveResult.rightCount == 0)
                desc << ". Keep right";
            else if (moveResult.rightCount > 0 && moveResult.centerCount == 0 && moveResult.leftCount == 0)
                desc << ". Keep left";

            std::string spoken = desc.str();

            {
                std::lock_guard<std::mutex> lk(descMutex_);
                lastDescription_ = spoken;
            }

            // Speak (skip trivial phrases; allow repeating after a cooldown)
            bool trivial = spoken.empty() || spoken == "Path clear";
            if (voiceEnabled_.load() && tts_ && !trivial) {
                // Decrement cooldown timer
                if (vaForceTimer_ > 0.0) {
                    vaForceTimer_ -= speakIntervalSec_;
                    if (vaForceTimer_ < 0.0) vaForceTimer_ = 0.0;
                }

                if (spoken != lastVADescription_ || vaForceTimer_ <= 0.0) {
                    lastVADescription_ = spoken;
                    vaForceTimer_      = 30.0; // seconds before we allow repeats
                    tts_->speak(spoken);
                    std::cout << "[VoiceAssist] " << spoken << "\n";
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (voiceEnabled_.load() && tts_) {
        tts_->speak("Voice assist mode off");
    }
}

} // namespace SmartGlasses

