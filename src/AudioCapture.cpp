#include "AudioCapture.h"

#include <fstream>
#include <mutex>

AudioCapture gAudioCapture;

static void logLineA(const std::string& s) {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    std::ofstream f("camera_log.txt", std::ios::out | std::ios::app);
    if (!f.is_open()) return;
    f << s << "\n";
}

#if __has_include(<portaudio.h>)
#  define HAVE_PORTAUDIO 1
#  include <portaudio.h>
#else
#  define HAVE_PORTAUDIO 0
#endif

#if __has_include(<samplerate.h>)
#  define USE_LIBSAMPLERATE 1
#  include <samplerate.h>
#else
#  define USE_LIBSAMPLERATE 0
#endif

#if HAVE_PORTAUDIO
// Adapter with exact PaStreamCallback signature for Pa_OpenStream; forwards to implementation.
static int paCallbackAdapter(const void* input, void* output, unsigned long frameCount,
                             const PaStreamCallbackTimeInfo* /*timeInfo*/, PaStreamCallbackFlags /*statusFlags*/,
                             void* userData) {
    return AudioCapture::paCallbackPublic(input, output, frameCount, nullptr, 0, userData);
}
#endif

AudioCapture::AudioCapture() {
    ring_.resize(RING_SIZE, 0.f);

#if HAVE_PORTAUDIO
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        logLineA(std::string("AudioCapture: Pa_Initialize FAILED — ") + Pa_GetErrorText(err));
        return;
    }

    int n = Pa_GetDeviceCount();
    if (n < 0) {
        logLineA(std::string("AudioCapture: Pa_GetDeviceCount FAILED — ") + Pa_GetErrorText(n));
        return;
    }
    logLineA("AudioCapture: devices=" + std::to_string(n));
    for (int i = 0; i < n; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info) continue;
        if (info->maxInputChannels <= 0) continue;
        logLineA(std::string("AudioCapture: input[") + std::to_string(i) + "] " + info->name);
    }
#else
    // STUB — uncomment PortAudio impl when lib is installed
    logLineA("AudioCapture: PortAudio headers not found — capture disabled (stub)");
#endif
}

AudioCapture::~AudioCapture() {
    stop();
}

bool AudioCapture::start() {
#if !HAVE_PORTAUDIO
    // STUB — uncomment PortAudio impl when lib is installed
    return false;
#else
    int deviceCount = Pa_GetDeviceCount();
    if (deviceCount <= 0) {
        logLineA("[Audio] No audio devices found");
        return false;
    }
    PaDeviceIndex dev = Pa_GetDefaultInputDevice();
    if (dev == paNoDevice) {
        logLineA("[Audio] No default input device");
        return false;
    }

    const PaDeviceInfo* info = Pa_GetDeviceInfo(dev);
    deviceName_ = info && info->name ? info->name : "Unknown";
    logLineA("[Audio] Using device: " + deviceName_);

    PaStreamParameters inParams{};
    inParams.device = dev;
    inParams.channelCount = 1;   // mono — Whisper expects 16kHz mono
    inParams.sampleFormat = paFloat32;
    inParams.suggestedLatency = info ? info->defaultLowInputLatency : 0.0;
    inParams.hostApiSpecificStreamInfo = nullptr;

    // 16000 Hz, 1024 frames per buffer — matches Whisper input
    const double sampleRate = 16000.0;
    const unsigned long framesPerBuf = 1024;

    PaError err = Pa_OpenStream(
        reinterpret_cast<PaStream**>(&stream_),
        &inParams,
        nullptr,
        sampleRate,
        framesPerBuf,
        paNoFlag,
        paCallbackAdapter,
        this);
    if (err != paNoError) {
        logLineA(std::string("AudioCapture: Pa_OpenStream FAILED — ") + Pa_GetErrorText(err));
        stream_ = nullptr;
        return false;
    }

    err = Pa_StartStream(reinterpret_cast<PaStream*>(stream_));
    if (err != paNoError) {
        logLineA(std::string("AudioCapture: Pa_StartStream FAILED — ") + Pa_GetErrorText(err));
        Pa_CloseStream(reinterpret_cast<PaStream*>(stream_));
        stream_ = nullptr;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(ringMtx_);
        readPos_ = 0;
        writePos_ = 0;
        available_ = 0;
    }
    running_ = true;
    return true;
#endif
}

int AudioCapture::paCallbackPublic(const void* input, void* output, unsigned long frameCount,
                                   const void* timeInfo, unsigned long statusFlags, void* userData) {
    return paCallback(input, output, frameCount, timeInfo, statusFlags, userData);
}

void AudioCapture::stop() {
#if !HAVE_PORTAUDIO
    running_ = false;
    ringCv_.notify_all();
    return;
#else
    running_.store(false);
    ringCv_.notify_all();

    if (stream_) {
        Pa_StopStream(reinterpret_cast<PaStream*>(stream_));
        Pa_CloseStream(reinterpret_cast<PaStream*>(stream_));
        stream_ = nullptr;
    }
    // Do NOT call Pa_Terminate() here — we need to be able to start() again for 2nd+ PTT.
    // PortAudio stays initialized for the app lifetime; only the stream is closed.
#endif
}

bool AudioCapture::read(std::vector<float>& outBuffer, int numSamples) {
#if !HAVE_PORTAUDIO
    // STUB — uncomment PortAudio impl when lib is installed
    (void)outBuffer;
    (void)numSamples;
    return false;
#else
    std::unique_lock<std::mutex> lock(ringMtx_);
    ringCv_.wait(lock, [&]{ return available_ >= numSamples || !running_; });
    if (!running_ && available_ < numSamples) return false;

    outBuffer.resize(numSamples);
    for (int i = 0; i < numSamples; ++i) {
        outBuffer[i] = ring_[readPos_];
        readPos_ = (readPos_ + 1) % RING_SIZE;
    }
    available_ -= numSamples;
    return true;
#endif
}

int AudioCapture::paCallback(const void* input, void* /*output*/,
                             unsigned long frameCount,
                             const void* /*timeInfo*/,
                             unsigned long /*statusFlags*/,
                             void* userData) {
#if !HAVE_PORTAUDIO
    (void)input; (void)frameCount; (void)userData;
    return 0;
#else
    auto* self = static_cast<AudioCapture*>(userData);
    if (!self || !self->running_) return paContinue;
    if (!input) return paContinue;

    const float* in = static_cast<const float*>(input);

    // Stream opened at 16kHz mono paFloat32 — push directly, no resampling
    self->pushSamples(in, static_cast<int>(frameCount));
    return paContinue;
#endif
}

void AudioCapture::pushSamples(const float* samples, int count) {
    std::lock_guard<std::mutex> lock(ringMtx_);
    for (int i = 0; i < count; ++i) {
        ring_[writePos_] = samples[i];
        writePos_ = (writePos_ + 1) % RING_SIZE;
        if (available_ < RING_SIZE) {
            ++available_;
        } else {
            // Overwrite oldest
            readPos_ = (readPos_ + 1) % RING_SIZE;
        }
    }
    ringCv_.notify_one();
}

