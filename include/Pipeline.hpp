#include <mutex>
#include <chrono>
#include <functional>
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include "face/face_encoder.h"
#include "face/identity_store.h"
#include "object/ObjectEncoder.hpp"
#include "object/ObjectRegistry.hpp"

namespace SmartGlasses {

enum class EnrollmentType { FACE, OBJECT };

class CameraManager;
class ModelRegistry;
class TTSService;

// MoveSafe result: (min(Di) > D_threshold) per PDF research.
struct MoveSafeResult {
    std::string hint;       // e.g. "Path clear", "Obstacle ahead - slow down"
    int leftCount = 0;
    int rightCount = 0;
    int centerCount = 0;
    double minDistanceNorm = 1.0;  // min normalized distance (1 = far, 0 = close)
    double maxProbability = 0.0;   // max detection probability
    std::vector<std::vector<float>> boxes;  // [x1,y1,x2,y2] for HUD overlay
};

// OCR result with confidence (0..1). Only speak if confidence > threshold.
struct OCRResult {
    std::string text;
    float confidence = 0.f;
};

// Pipeline: runs vision models and routes outputs to TTS.
class Pipeline {
public:
    Pipeline(CameraManager& camera, ModelRegistry& registry, TTSService& tts);

    // Object detection -> MoveSafe decision -> TTS. Returns hint and draws boxes on frame.
    MoveSafeResult runObstacleDetection(cv::Mat& frame, bool speak = true);

    // OCR -> format text -> speak only if confidence > 60%. Returns text and confidence.
    OCRResult runOCR(cv::Mat& frame, bool speak = true);

    // Face detection -> optional TTS (e.g. "Person detected").
    void runFaceRecognition(cv::Mat& frame, std::vector<cv::Rect>& faces, bool speak = true);

    // Sign placeholder: can be extended with gesture model; route text -> TTS.
    std::string runSignToText(const cv::Mat& frame);

    // OCR Google Lens: live scan (~1s) and manual capture. Reads/writes ResultsStore.
    void tickOCR(const cv::Mat& frame, std::chrono::steady_clock::time_point now);

    // Enrollment: grab 50 frames, crop bbox[targetIdx], call progressCb(i/50.f) every 5 frames, doneCb(success).
    void captureEnrollmentFrames(int targetIdx, const std::string& label, EnrollmentType type,
                                 std::function<void(float)> progressCb,
                                 std::function<void(bool)> doneCb);
    void getLastEnrollmentFrames(std::vector<cv::Mat>& out) const;

    static std::string formatOCRText(const std::string& raw);

private:
    // TTS throttle (thread-safe member state)
    std::string                            lastSpoken_;
    std::chrono::steady_clock::time_point  lastSpeakTime_;
    bool                                   firstSpeak_       = true;
    std::mutex                             ttsMutex_;
    // State-change trackers
    std::string                            lastObstacleZone_;
    int                                    lastFaceCount_    = -1;
    // Internal helper
    bool shouldSpeak(const std::string& phrase, float cooldownSec = 4.0f);

    static bool isObstacleClass(int classId);

    CameraManager* camera_ = nullptr;
    ModelRegistry* registry_ = nullptr;
    TTSService* tts_ = nullptr;

    mutable std::mutex enrollmentMutex_;
    std::vector<cv::Mat> lastEnrollmentFrames_;

    // Face and object enrollment / recognition
    std::unique_ptr<hud::FaceEncoder>    faceEncoder_;
    std::unique_ptr<hud::IdentityStore>  faceStore_;
    std::unique_ptr<hud::ObjectEncoder>  objectEncoder_;
    std::unique_ptr<hud::ObjectRegistry> objectRegistry_;
    float                                faceMatchThreshold_   = 0.6f;
    float                                objectMatchThreshold_ = 0.7f;
};

} // namespace SmartGlasses
