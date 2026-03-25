#include <atomic>
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <filesystem>
#include <cstdint>

#ifdef HAS_WHISPER
#  include <whisper.h>
#endif

#if __has_include(<portaudio.h>)
#  include <portaudio.h>
#endif

#include "AudioCapture.h"
#include "Config.hpp"
#include "ResultsStore.h"
#include "STTEngine.h"
#include "TranslationQueue.h"
#include "PythonBridge.hpp"
#include <iostream>

extern AudioCapture gAudioCapture;

static const int kSTTSampleRate = 16000;
static const int kSTTFramesPerBuffer = 512;
static const float kSTTMaxRecordSec = 30.0f;
static const int kPushToTalkChunkSamples = 800;  // ~50 ms at 16 kHz

namespace fs = std::filesystem;

static bool writeWav16(const std::string& path, const std::vector<float>& samples, int sampleRate) {
    if (samples.empty()) return false;
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const int numSamples = static_cast<int>(samples.size());
    const int numBytes = numSamples * 2;
    const int dataSize = numBytes;
    const int fileSize = 36 + dataSize;
    // RIFF header
    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&fileSize), 4);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    const int32_t fmtLen = 16;
    out.write(reinterpret_cast<const char*>(&fmtLen), 4);
    const int16_t audioFormat = 1;
    out.write(reinterpret_cast<const char*>(&audioFormat), 2);
    const int16_t numChannels = 1;
    out.write(reinterpret_cast<const char*>(&numChannels), 2);
    out.write(reinterpret_cast<const char*>(&sampleRate), 4);
    const int32_t byteRate = sampleRate * 2;
    out.write(reinterpret_cast<const char*>(&byteRate), 4);
    const int16_t blockAlign = 2;
    out.write(reinterpret_cast<const char*>(&blockAlign), 2);
    const int16_t bitsPerSample = 16;
    out.write(reinterpret_cast<const char*>(&bitsPerSample), 2);
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&dataSize), 4);
    for (int i = 0; i < numSamples; ++i) {
        float s = std::max(-1.f, std::min(1.f, samples[i]));
        int16_t v = static_cast<int16_t>(s * 32767.f);
        out.write(reinterpret_cast<const char*>(&v), 2);
    }
    return out.good();
}

static void STTLog(const std::string& msg) {
    std::ofstream f("hud_log.txt", std::ios::out | std::ios::app);
    if (f.is_open()) { f << msg << "\n"; f.flush(); }
    std::cout << msg << "\n";
}

STTEngine gSTTEngine;

namespace SmartGlasses {
std::string asyncSTT(int durationSec) {
    return PythonBridge::runSTT(durationSec);
}
}

STTEngine::STTEngine() = default;

STTEngine::~STTEngine() {
    stop();
}

bool STTEngine::start() {
    // Background Whisper engine is no longer used; STT is now handled
    // on-demand via PythonBridge. Report ready so HUD shows it as available.
    hasWhisper_   = false;
    activeEngine_ = Engine::NONE;
    gResultsStore.setSTTStatus("ready");
    return true;
}

void STTEngine::stop() {
    if (!running_) return;
    running_ = false;
    gResultsStore.setSTTStatus("disabled");
    if (thread_.joinable()) thread_.join();
}

std::string STTEngine::statusString() const {
    return "Python STT (external)";
}

void STTEngine::run() {
    // Background continuous STT loop is no longer used; STT is handled
    // via startListening()/recordLoop calling the Python bridge.
    gResultsStore.setSTTActive(false);
    gResultsStore.setSTTStatus("disabled");
}

void STTEngine::startListening(bool pushToTalk) {
    if (isRecording_.load()) return;
    STTLog(std::string("[STT] startListening() pushToTalk=") + (pushToTalk ? "true" : "false"));
    gResultsStore.setSTTOriginal("", "en");
    gResultsStore.setSTTPartial(pushToTalk ? "Recording... (hold Space)" : "Recording... 0s");
    gResultsStore.setSTTActive(true);
    gResultsStore.setSTTStatus("listening");
    isRecording_.store(true);
    recordBuffer_.clear();
    if (pushToTalk) {
        pushToTalkMode_.store(true);
        if (!gAudioCapture.isRunning() && !gAudioCapture.start()) {
            gResultsStore.setSTTActive(false);
            gResultsStore.setSTTPartial("Microphone unavailable");
            gResultsStore.setSTTStatus("ready");
            isRecording_.store(false);
            pushToTalkMode_.store(false);
            return;
        }
        recordThread_ = std::thread(&STTEngine::recordLoopPushToTalk, this);
    } else {
        recordThread_ = std::thread(&STTEngine::recordLoop, this);
    }
    recordThread_.detach();
}

void STTEngine::stopListening() {
    isRecording_.store(false);
    if (pushToTalkMode_.load())
        gAudioCapture.stop();
    STTLog("[STT] stopListening() called");
}

void STTEngine::recordLoop() {
    static std::atomic<bool> s_sttBusy{false};
    if (s_sttBusy.exchange(true)) return;  // prevent overlapping STT calls
    struct Guard { std::atomic<bool>& b; ~Guard(){ b=false; } } g{s_sttBusy};
    gResultsStore.setSTTPartial("Recording...");
    gResultsStore.setSTTActive(true);
    gResultsStore.setSTTStatus("listening");

    std::string text = SmartGlasses::PythonBridge::runSTT(static_cast<int>(kSTTMaxRecordSec));

    gResultsStore.setSTTActive(false);
    std::string result = text.empty() ? "[Nothing heard]" : text;
    gResultsStore.setSTTOriginal(result, "en");
    gResultsStore.setSTTPartial("");
    gResultsStore.setSTTStatus("ready");
    STTLog("[STT] On-demand result: " + result);
}

void STTEngine::recordLoopPushToTalk() {
    static std::atomic<bool> s_pttBusy{false};
    if (s_pttBusy.exchange(true)) {
        pushToTalkMode_.store(false);
        return;
    }
    std::vector<float> toTranscribe;
    {
        struct Guard { std::atomic<bool>& b; ~Guard(){ b=false; } } g{s_pttBusy};
        std::vector<float> buffer;
        buffer.reserve(kPushToTalkChunkSamples * 2);
        while (isRecording_.load()) {
            std::vector<float> chunk;
            if (!gAudioCapture.read(chunk, kPushToTalkChunkSamples))
                break;
            std::lock_guard<std::mutex> lock(recordMutex_);
            recordBuffer_.insert(recordBuffer_.end(), chunk.begin(), chunk.end());
        }
        gAudioCapture.stop();
        pushToTalkMode_.store(false);
        {
            std::lock_guard<std::mutex> lock(recordMutex_);
            toTranscribe.swap(recordBuffer_);
        }
        gResultsStore.setSTTActive(false);
        // Guard destroyed here – s_pttBusy = false so next Space press can start immediately
    }

    std::string result;
    const int minSamples = kSTTSampleRate / 10;  // at least 0.1 s
    if (toTranscribe.size() < static_cast<size_t>(minSamples)) {
        result = "[Nothing heard]";
    } else {
        std::string wavPath;
        try {
            wavPath = (fs::temp_directory_path() / "stt_push.wav").string();
        } catch (...) {
            wavPath = "stt_push.wav";
        }
        if (!writeWav16(wavPath, toTranscribe, kSTTSampleRate)) {
            result = "[Recording failed]";
        } else {
            result = SmartGlasses::PythonBridge::runSTTFromFile(wavPath);
            if (result.empty()) result = "[Nothing heard]";
            try { fs::remove(wavPath); } catch (...) {}
        }
    }
    gResultsStore.setSTTOriginal(result, "en");
    gResultsStore.setSTTPartial("");
    gResultsStore.setSTTStatus("ready");
    STTLog("[STT] Push-to-talk result: " + result);
}

bool STTEngine::detectVoiceActivity(const std::vector<float>& samples) const {
    if (samples.empty()) return false;
    double sum = 0.0;
    for (float s : samples) sum += static_cast<double>(s) * s;
    float rms = std::sqrt(static_cast<float>(sum / samples.size()));
    return rms > SmartGlasses::Config::VAD_ENERGY_THRESHOLD;
}

std::string STTEngine::transcribe(const std::vector<float>& samples) {
#ifdef HAS_WHISPER
    if (!ctx_) return "";

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language          = "en";
    params.translate         = false;
    params.n_threads         = 4;
    params.print_progress    = false;
    params.print_realtime    = false;
    params.print_timestamps  = false;
#if defined(whisper_full_params_has_no_context)
    params.no_context        = true;
#endif
#if defined(whisper_full_params_has_single_segment)
    params.single_segment    = false;
#endif
#if defined(whisper_full_params_has_suppress_blank)
    params.suppress_blank    = true;
#endif
#if defined(whisper_full_params_has_suppress_non_speech_tokens)
    params.suppress_non_speech_tokens = true;
#endif
#if defined(whisper_full_params_has_temperature)
    params.temperature       = 0.0f;
#endif
#if defined(whisper_full_params_has_temperature_inc)
    params.temperature_inc   = 0.0f;
#endif
#if defined(whisper_full_params_has_greedy) && defined(whisper_greedy_params_has_best_of)
    params.greedy.best_of    = 1;
#endif

    if (whisper_full(ctx_, params, samples.data(), static_cast<int>(samples.size())) != 0)
        return "";

    std::string result;
    int nSeg = whisper_full_n_segments(ctx_);
    for (int i = 0; i < nSeg; ++i) {
        const char* segPtr = whisper_full_get_segment_text(ctx_, i);
        if (!segPtr) continue;
        std::string seg = segPtr;
        if (seg.find("[BLANK_AUDIO]") != std::string::npos) continue;
        if (seg.find("(music)") != std::string::npos) continue;
        if (seg.find("Thank you") != std::string::npos && nSeg == 1) continue;
        if (seg.find("you") != std::string::npos && seg.size() < 5) continue;
        if (seg.find("Subscribe") != std::string::npos) continue;
        if (seg.find("thanks for watching") != std::string::npos) continue;
        if (seg.find("bye") != std::string::npos && seg.size() < 6) continue;
        if (seg.find("the") != std::string::npos && seg.size() < 6) continue;
        if (seg.find("and") != std::string::npos && seg.size() < 6) continue;
        if (seg.find("to") != std::string::npos && seg.size() < 5) continue;
        while (!seg.empty() && (seg.front() == ' ' || seg.front() == '.'))
            seg.erase(seg.begin());
        while (!seg.empty() && (seg.back() == ' ' || seg.back() == '.'))
            seg.pop_back();
        if (seg.size() >= 3) result += seg + " ";
    }
    if (!result.empty()) result = result.substr(0, result.size() - 1);
    if (result.size() < 3) return "";
    if (result.find("...") != std::string::npos && result.size() < 10) return "";
    if (!result.empty()) STTLog("[STT] Recognized: " + result);
    return result;
#else
    (void)samples;
    return "";
#endif
}

std::string STTEngine::detectLanguage(const std::string& text) const {
    bool hasJa = false, hasZh = false, hasKo = false, hasAr = false;

    const unsigned char* p   = reinterpret_cast<const unsigned char*>(text.c_str());
    const unsigned char* end = p + text.size();

    while (p < end) {
        uint32_t cp = 0;
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0 && p + 1 < end) {
            cp = ((*p & 0x1F) << 6) | (p[1] & 0x3F);
            p += 2;
        } else if ((*p & 0xF0) == 0xE0 && p + 2 < end) {
            cp = ((*p & 0x0F) << 12) |
                 ((p[1] & 0x3F) << 6) |
                 (p[2] & 0x3F);
            p += 3;
        } else {
            ++p;
            continue;
        }

        if (cp >= 0x3040 && cp <= 0x30FF) hasJa = true;
        if (cp >= 0x4E00 && cp <= 0x9FFF) hasZh = true;
        if (cp >= 0xAC00 && cp <= 0xD7AF) hasKo = true;
        if (cp >= 0x0600 && cp <= 0x06FF) hasAr = true;
    }

    if (hasJa) return "ja";
    if (hasKo) return "ko";
    if (hasAr) return "ar";
    if (hasZh) return "zh";
    return "auto";
}

