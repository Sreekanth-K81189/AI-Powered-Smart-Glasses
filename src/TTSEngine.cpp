#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>

#include "Config.hpp"
#include "ResultsStore.h"
#include "TTSEngine.h"
#include "TTSQueue.h"

#ifdef HAS_PORTAUDIO
#  include <portaudio.h>
#endif

TTSEngine gTTSEngine;

TTSEngine::TTSEngine() = default;

TTSEngine::~TTSEngine() {
    stop();
}

bool TTSEngine::start() {
    using SmartGlasses::Config;
    hasPiper_ = std::filesystem::exists(Config::piperExePath) &&
                std::filesystem::exists(Config::piperModelPath);

    running_ = true;
    gTTSQueue.reset();
    thread_ = std::thread([this]{ run(); });
    return true;
}

void TTSEngine::stop() {
    if (!running_) return;
    running_ = false;
    gTTSQueue.stop();
    if (thread_.joinable()) thread_.join();
}

std::string TTSEngine::statusString() const {
    if (!running_)  return "disabled";
    if (!hasPiper_) return "active (no piper)";
    return "Piper/lessac";
}

void TTSEngine::run() {
    using SmartGlasses::Config;

    while (running_) {
        std::string text;
        if (!gTTSQueue.waitAndPop(text)) break;
        if (!Config::ttsEnabled || text.empty()) continue;

        gResultsStore.setTTSSpokenText(text);
        gResultsStore.setTTSPlaying(true);

        if (hasPiper_) {
            auto samples = synthesise(text);
            if (!samples.empty())
                playAudio(samples, Config::PIPER_SAMPLE_RATE);
        } else {
            // Simulate duration so HUD indicator is visible
            std::this_thread::sleep_for(std::chrono::milliseconds(
                std::min<int>(2000, std::max<int>(500, (int)text.size() * 20))));
        }

        gResultsStore.setTTSPlaying(false);
    }
    gResultsStore.setTTSPlaying(false);
}

std::vector<float> TTSEngine::synthesise(const std::string& text) {
    using SmartGlasses::Config;

    std::vector<float> out;
    if (!hasPiper_) return out;

#ifdef _WIN32
    // Write Piper output to a temporary raw file (int16 PCM)
    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "piper_out.raw";
    std::string tmpPath = tmp.string();

    std::string cmd = "echo " + text + " | \""
        + Config::piperExePath + "\" --model \"" + Config::piperModelPath
        + "\" --output_raw > \"" + tmpPath + "\" 2>NUL";

    std::system(cmd.c_str());

    FILE* f = std::fopen(tmpPath.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        std::fclose(f);
        std::filesystem::remove(tmp);
        return out;
    }
    std::vector<int16_t> pcm((size_t)sz / sizeof(int16_t));
    std::fread(pcm.data(), sizeof(int16_t), pcm.size(), f);
    std::fclose(f);
    std::filesystem::remove(tmp);

    out.reserve(pcm.size());
    for (int16_t s : pcm) {
        float v = static_cast<float>(s) / 32768.f;
        v *= Config::ttsVolume;
        out.push_back(v);
    }
#endif
    return out;
}

void TTSEngine::playAudio(std::vector<float>& samples, int sampleRate) {
#ifdef HAS_PORTAUDIO
    PaStream* stream = nullptr;
    if (Pa_OpenDefaultStream(&stream, 0, 1, paFloat32,
                             sampleRate, 256, nullptr, nullptr) != paNoError)
        return;

    Pa_StartStream(stream);

    const int chunk = 256;
    int offset = 0;
    while (offset < (int)samples.size() && running_) {
        int frames = std::min(chunk, (int)samples.size() - offset);
        Pa_WriteStream(stream, samples.data() + offset, frames);
        offset += frames;
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
#else
    (void)samples;
    (void)sampleRate;
#endif
}

