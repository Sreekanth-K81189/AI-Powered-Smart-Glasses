#pragma once
#include <mutex>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <opencv2/core.hpp>

// ── YOLO detection box (normalised 0-1 against capture dimensions) ──────────
struct DetectionBox {
    float x1, y1, x2, y2;   // normalised against Config::CAPTURE_WIDTH/HEIGHT
    int   classId;
    float confidence;
    std::string label;
};

// ── Face recognition result (in capture pixels 1280×720) ────────────────────
struct FaceResult {
    cv::Rect    rect;         // pixel rect in capture space
    std::string name;
    float       confidence;
};

// ── OCR text region (Google Lens style: bbox + optional recognized text) ─────
struct OCRRegion {
    cv::Rect    bbox;
    std::string text;
    float       confidence = 0.f;
    bool        selected   = false;
    float       highlightTimer = 0.f;
};

// ── Single translation result ────────────────────────────────────────────────
struct TranslationResult {
    std::string source;       // "OCR" or "STT"
    std::string original;     // raw text before translation
    std::string translated;   // translated text in target language
    std::string sourceLang;   // detected source language code e.g. "ja"
    int64_t     timestamp;    // ms since epoch
};

// ── Snapshot struct — HUDLayer reads this, never the store directly ──────────
struct HUDSnapshot {
    // YOLO
    std::vector<DetectionBox> boxes;

    // Face
    std::vector<FaceResult>   faces;

    // STT
    std::string sttOriginal;
    std::string sttTranslated;
    std::string sttPartial;   // live status: "Recording... Xs" or "Transcribing..."
    int64_t     sttTimestamp  = 0;
    bool        sttActive     = false;   // true when STT is currently listening

    // OCR
    std::string ocrOriginal;
    std::string ocrTranslated;
    bool        ocrVisible    = false;
    int64_t     ocrTimestamp  = 0;
    float       ocrConfidence = 0.f;     // 0..100 for display
    std::vector<OCRRegion> ocrRegions;
    std::vector<cv::RotatedRect> ocrBoxesRotated;  // EAST boxes for overlay
    int         selectedOCRRegion = -1;
    float       ocrOverlayTimer   = 0.f;
    std::string selectedOCRText;   // text for floating overlay at selected region
    // OCR Google Lens style state
    bool        ocrLiveOn         = false;
    bool        ocrCropMode       = false;
    bool        ocrCaptureRequested = false;
    bool        ocrImageCaptured  = false;  // true after 1st tap = "Tap to read"
    bool        ocrUseCrop         = false;
    cv::Rect    ocrCropRegion;

    // TTS
    bool        ttsPlaying    = false;
    std::string ttsSpokenText;   // text currently being spoken (for lower overlay)

    // Feed info (for Settings panel display)
    int captureW   = 1280;
    int captureH   = 720;
    int detectionW = 640;
    int detectionH = 480;

    // SAPI availability (STT/TTS on Windows)
    bool sapiAvailable = true;   // assume available until proven otherwise

    // Module status strings for Settings panel
    std::string sttStatus;   // "disabled" / "listening" / "active" / "error"
    std::string ocrStatus;   // "disabled" / "ready" / "processing" / "error"
    std::string ttsStatus;   // "disabled" / "idle" / "speaking" / "error"
};

// ── ResultsStore ─────────────────────────────────────────────────────────────
class ResultsStore {
public:
    // ── Thread write methods (called from background threads) ─────────────

    // YOLO thread writes new detection boxes
    void setBoxes(std::vector<DetectionBox> boxes);

    // Face thread writes new face results
    void setFaces(std::vector<FaceResult> faces);

    // STT thread writes new transcription (before translation)
    void setSTTOriginal(const std::string& text, const std::string& lang);

    // TranslationEngine writes completed STT translation
    void setSTTTranslated(const std::string& text);

    // STT thread sets listening state
    void setSTTActive(bool active);

    // OCR thread writes new OCR text (before translation)
    void setOCROriginal(const std::string& text);

    // TranslationEngine writes completed OCR translation
    void setOCRTranslated(const std::string& text, bool visible);

    // TTS thread sets playback state
    void setTTSPlaying(bool playing);
    void setTTSSpokenText(const std::string& text);

    // Pipeline sets feed dimensions on startup
    void setFeedInfo(int captureW, int captureH, int detW, int detH);

    // SAPI availability (set by SpeechService/TTSService on Windows)
    void setSAPIAvailable(bool available);

    void setSTTStatus(const std::string& status);
    void setSTTPartial(const std::string& partial);  // "Recording... Xs" or "Transcribing..."
    void setOCRStatus(const std::string& status);
    void setTTSStatus(const std::string& status);

    // OCR Google Lens style
    void setOCRLiveOn(bool on);
    bool getOCRLiveOn() const;
    void setOCRCropMode(bool on);
    bool getOCRCropMode() const;
    void setOCRCaptureRequested(bool req);
    bool getOCRCaptureRequested() const;
    void setOCRImageCaptured(bool captured);
    bool getOCRImageCaptured() const;
    void setOCRUseCrop(bool use);
    void setOCRCropRegion(cv::Rect r);
    cv::Rect getOCRCropRegion() const;
    void setLastOcrScanTime(std::chrono::steady_clock::time_point t);
    std::chrono::steady_clock::time_point getLastOcrScanTime() const;
    void setLastOcrSpeakTime(std::chrono::steady_clock::time_point t);
    std::chrono::steady_clock::time_point getLastOcrSpeakTime() const;
    void setLastSpokenOcr(const std::string& s);
    std::string getLastSpokenOcr() const;
    void setOCRBoxesRotated(const std::vector<cv::RotatedRect>& boxes);
    void setOCRConfidence(float c);
    float getOCRConfidence() const;

    void setOCRRegions(const std::vector<OCRRegion>& regions);
    void setSelectedOCRRegion(int index, const std::string& text);
    void tickOCRTimers(float dt);

    void setSignWord(const std::string& word);
    std::string getSignWord() const;
    void setSignConfidence(float confidence);
    float getSignConfidence() const;

    // ── HUDLayer read method (called from main/render thread only) ────────

    // Returns a complete snapshot — lock held only during copy, not during render
    HUDSnapshot snapshot() const;

    // ── Utility ───────────────────────────────────────────────────────────

    // Clears all detection results (called when camera disconnects)
    void clear();

private:
    mutable std::mutex mtx_;

    std::vector<DetectionBox> boxes_;
    std::vector<FaceResult>   faces_;

    std::string sttOriginal_;
    std::string sttTranslated_;
    std::string sttPartial_;
    int64_t     sttTimestamp_  = 0;
    bool        sttActive_     = false;

    std::string ocrOriginal_;
    std::string ocrTranslated_;
    bool        ocrVisible_    = false;
    int64_t     ocrTimestamp_  = 0;
    float       ocrConfidence_ = 0.f;
    std::vector<cv::RotatedRect> ocrBoxesRotated_;
    bool        ocrLiveOn_         = false;
    bool        ocrCropMode_       = false;
    bool        ocrCaptureRequested_ = false;
    bool        ocrImageCaptured_    = false;
    bool        ocrUseCrop_        = false;
    cv::Rect    ocrCropRegion_;
    std::chrono::steady_clock::time_point lastOcrScanTime_{};
    std::chrono::steady_clock::time_point lastOcrSpeakTime_{};
    std::string lastSpokenOcr_;

    bool        ttsPlaying_    = false;
    std::string ttsSpokenText_;

    int captureW_   = 1280;
    int captureH_   = 720;
    int detectionW_ = 640;
    int detectionH_ = 480;
    bool sapiAvailable_ = true;

    std::string sttStatus_ = "disabled";
    std::string ocrStatus_ = "disabled";
    std::string ttsStatus_ = "disabled";

    std::vector<OCRRegion> ocrRegions_;
    int         selectedOCRRegion_ = -1;
    float       ocrOverlayTimer_   = 0.f;
    std::string selectedOCRText_;

    std::string signWord_;
    float       signConfidence_ = 0.f;
};

extern ResultsStore gResultsStore;

