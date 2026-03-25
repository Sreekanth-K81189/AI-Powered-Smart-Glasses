#include "speech/tts_engine.h"
#include <iostream>
#include <chrono>
#include <filesystem>

// Windows SAPI
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif
#include <sapi.h>
#pragma comment(lib, "ole32.lib")

namespace hud {

TtsEngine::TtsEngine() {
    // Check for eSpeak-NG
    for (auto& p : {"C:\\Program Files\\eSpeak NG\\espeak-ng.exe",
                    "C:\\Program Files (x86)\\eSpeak NG\\espeak-ng.exe"}) {
        if (std::filesystem::exists(p)) {
            eSpeakPath_ = p;
            useESpeak_  = true;
            break;
        }
    }
    std::cout << "[TTS] Using " << (useESpeak_ ? "eSpeak-NG" : "Windows SAPI") << "\n";
    CoInitialize(nullptr);
    worker_ = std::thread(&TtsEngine::WorkerLoop, this);
}

TtsEngine::~TtsEngine() {
    running_ = false;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    CoUninitialize();
}

void TtsEngine::Speak(const std::string& text, int priority) {
    std::lock_guard<std::mutex> lk(mtx_);
    queue_.push({text, priority});
    cv_.notify_one();
}

void TtsEngine::SpeakImmediate(const std::string& text) {
    std::lock_guard<std::mutex> lk(mtx_);
    while (!queue_.empty()) queue_.pop();
    queue_.push({text, 99});
    cv_.notify_one();
}

void TtsEngine::Stop() {
    std::lock_guard<std::mutex> lk(mtx_);
    while (!queue_.empty()) queue_.pop();
}

void TtsEngine::WorkerLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [this]{ return !queue_.empty() || !running_; });
        if (!running_) break;
        auto item = queue_.front(); queue_.pop();
        lk.unlock();

        // Throttle
        if (item.text == lastSpoken_) {
            auto now = (float)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0f;
            if (now - lastSpokenTime_ < throttleSec_) continue;
        }

        speaking_ = true;
        if (useESpeak_) SpeakESpeak(item.text);
        else            SpeakSAPI(item.text);
        speaking_       = false;
        lastSpoken_     = item.text;
        lastSpokenTime_ = (float)std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0f;
    }
}

void TtsEngine::SpeakSAPI(const std::string& text) {
    ISpVoice* pVoice = nullptr;
    if (FAILED(CoCreateInstance(CLSID_SpVoice, nullptr, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice)))
        return;
    std::wstring wtext(text.begin(), text.end());
    pVoice->SetRate(rate_ - 5);
    pVoice->SetVolume((USHORT)(volume_ * 100));
    pVoice->Speak(wtext.c_str(), SPF_DEFAULT, nullptr);
    pVoice->Release();
}

void TtsEngine::SpeakESpeak(const std::string& text) {
    std::string cmd = "\"" + eSpeakPath_ + "\" -s " + std::to_string(rate_) +
                      " \"" + text + "\" 2>nul";
    system(cmd.c_str());
}

void TtsEngine::SetRate(int rate)      { rate_   = rate;   }
void TtsEngine::SetVolume(float v)     { volume_ = v;      }

} // namespace hud
