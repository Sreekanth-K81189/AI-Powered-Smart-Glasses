#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <cstdint>
#include <string>

// Forward declare PortAudio types to avoid including portaudio.h in header
typedef void PaStream;

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();

    // Open PortAudio stream and start capture thread
    // Returns true on success, false if no microphone found
    bool start();

    // Stop capture and close PortAudio stream
    void stop();

    // Read exactly numSamples float32 mono 16kHz samples into outBuffer
    // BLOCKS until enough samples are available or stop() is called
    // Returns false if stop() was called (caller should exit)
    bool read(std::vector<float>& outBuffer, int numSamples);

    // Returns true if audio capture is active
    bool isRunning() const { return running_; }

    // Returns the detected input device name (for logging)
    std::string deviceName() const { return deviceName_; }

    // Called by PortAudio adapter only; same signature as paCallback for forwarding.
    static int paCallbackPublic(const void* input, void* output, unsigned long frameCount,
                                const void* timeInfo, unsigned long statusFlags, void* userData);

private:
    // PortAudio stream callback — called from PortAudio thread
    // Converts input samples to float32 mono 16kHz and pushes to ring buffer
    static int paCallback(const void* input, void* output,
                          unsigned long frameCount,
                          const void* timeInfo,
                          unsigned long statusFlags,
                          void* userData);

    // Internal: push samples into ring buffer (called from paCallback)
    void pushSamples(const float* samples, int count);

    PaStream*               stream_     = nullptr;
    std::atomic<bool>       running_    {false};

    // Ring buffer
    static constexpr int    RING_SIZE   = 16000 * 10;  // 10 sec at 16kHz
    std::vector<float>      ring_;
    int                     writePos_   = 0;
    int                     readPos_    = 0;
    int                     available_  = 0;            // samples ready to read
    std::mutex              ringMtx_;
    std::condition_variable ringCv_;

    std::string             deviceName_;
};

extern AudioCapture gAudioCapture;

