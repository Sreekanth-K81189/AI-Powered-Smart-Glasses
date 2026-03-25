#include "speech/stt_engine.h"
#include "whisper.h"
#include <portaudio.h>
#include <iostream>
#include <vector>
#include <cstring>

namespace hud {

SttEngine::SttEngine(const std::string& modelPath) : modelPath_(modelPath) {
    whisper_context_params cparams = whisper_context_default_params();
    whisperCtx_ = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!whisperCtx_)
        std::cerr << "[STT] Failed to load Whisper model: " << modelPath << "\n";
    else
        std::cout << "[STT] Whisper loaded: " << modelPath << "\n";
    Pa_Initialize();
}

SttEngine::~SttEngine() {
    Stop();
    if (whisperCtx_) whisper_free((whisper_context*)whisperCtx_);
    Pa_Terminate();
}

void SttEngine::Start(SttCallback cb) {
    callback_  = cb;
    listening_ = true;
    running_   = true;
    worker_    = std::thread(&SttEngine::WorkerLoop, this);
}

void SttEngine::Stop() {
    running_   = false;
    listening_ = false;
    if (worker_.joinable()) worker_.join();
}

void SttEngine::WorkerLoop() {
    while (running_) {
        auto audio = CaptureAudio(3000);
        if (audio.empty()) continue;
        auto text = RunWhisper(audio);
        if (!text.empty() && callback_) callback_(text);
    }
}

std::vector<float> SttEngine::CaptureAudio(int durationMs) {
    const int SAMPLE_RATE = 16000;
    int numSamples = (SAMPLE_RATE * durationMs) / 1000;
    std::vector<float> buffer(numSamples, 0.0f);

    PaStream* stream = nullptr;
    PaError err = Pa_OpenDefaultStream(&stream, 1, 0, paFloat32, SAMPLE_RATE, 256, nullptr, nullptr);
    if (err != paNoError) return {};

    Pa_StartStream(stream);
    int read = 0;
    while (read < numSamples && running_) {
        int toRead = std::min(256, numSamples - read);
        Pa_ReadStream(stream, buffer.data() + read, toRead);
        read += toRead;
    }
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    return buffer;
}

std::string SttEngine::RunWhisper(const std::vector<float>& audio) {
    if (!whisperCtx_ || audio.empty()) return "";
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language   = "en";
    params.print_realtime  = false;
    params.print_progress  = false;
    params.print_timestamps= false;
    int ret = whisper_full((whisper_context*)whisperCtx_, params, audio.data(), (int)audio.size());
    if (ret != 0) return "";
    std::string result;
    int nSeg = whisper_full_n_segments((whisper_context*)whisperCtx_);
    for (int i = 0; i < nSeg; i++)
        result += whisper_full_get_segment_text((whisper_context*)whisperCtx_, i);
    return result;
}

std::string SttEngine::TranscribeFile(const std::string& wavPath) {
    // Load wav, run whisper
    return RunWhisper(CaptureAudio(5000)); // simplified
}

} // namespace hud
