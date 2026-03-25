#pragma once
#include <string>
#include <algorithm>

namespace SmartGlasses {
    class Config {
    public:
        static std::string camera_url;
        static std::string modelsDir;
        static std::string tessDataPrefix;
        // Detection resolution for YOLO + face pipeline (0 = use capture size).
        static int cameraWidth;
        static int cameraHeight;
        // Optional separate resolution for downstream detectors / HUD.
        // Defaults to a mid-range value and can be overridden from the settings panel.
        inline static int detectionWidth  = 960;
        inline static int detectionHeight = 540;

        // Fixed base capture resolution requested from the camera.
        // The HUD always renders this full-resolution frame as the background.
        static constexpr int CAPTURE_WIDTH  = 1280;
        static constexpr int CAPTURE_HEIGHT = 720;

        // Camera and pipeline constants
        static constexpr double CAMERA_PROBE_TIMEOUT_SEC = 1.5; // seconds
        static const int USB_VIDEO_START = 0;
        static const int USB_VIDEO_END = 5;
        static const int USB_CAMERA_PROBE_MAX = 5;
        static constexpr float MOVESAFE_D_THRESHOLD     = 0.5f;
        static constexpr float OCR_CONFIDENCE_THRESHOLD = 0.60f; // 0..1 fraction

        // Feature toggles
        static bool cudaEnabled;
        static bool TTS_ENABLED; // legacy runtime toggle
        inline static bool ocrEnabled   = true;

        // HUD navigation
        static constexpr int   TOTAL_HUD_LAYERS = 5;
        static constexpr float HUD_MOVE_STEP    = 0.05f;
        static float hudPositionY;   // -1.0 top, +1.0 bottom
        static int   activeLayer;    // 0..TOTAL_HUD_LAYERS-1

        static float clampHudY(float v) {
            return std::clamp(v, -1.0f, 1.0f);
        }

        // STT
        inline static std::string whisperModelPath  = "models/ggml-base.en.bin";
        static constexpr int      AUDIO_SAMPLE_RATE = 16000;
        static constexpr int      AUDIO_CHUNK_MS    = 1500;
        static constexpr float    VAD_ENERGY_THRESHOLD = 0.02f;
        inline static bool        sttEnabled        = true;

        // TTS
        inline static std::string piperExePath      = "tts/piper/piper.exe";
        inline static std::string piperModelPath    = "tts/en_US-lessac-medium.onnx";
        static constexpr int      PIPER_SAMPLE_RATE = 22050;
        inline static bool        ttsEnabled        = true;
        inline static float       ttsVolume         = 1.0f;

        // Translation
        inline static std::string targetLanguage    = "en";
        // Output mode for merged OCR/Sign tasks: false = text only, true = speak
        inline static bool        ocrOutputToSpeech  = false;
        inline static bool        signOutputToSpeech = false;

        // Diagnostics
        inline static std::string gpuProvider       = "CUDA";
    };
}
