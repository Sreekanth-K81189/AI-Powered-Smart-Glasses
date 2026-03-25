/*
 * Pipeline.cpp  â€“  FIXED
 *
 * Root cause of "not detecting anything":
 *   runYOLO() returns pixel-coordinate boxes (x1,y1,x2,y2).
 *   runObstacleDetection() was treating those values as normalised 0-1,
 *   so:
 *     cx = ~640  â†’  never < 0.33, always > 0.66  â†’  leftCount=0, centerCount=0
 *     cy = ~360  â†’  dist = 1 - 360 = -359         â†’  always "very close"
 *   Result: "Obstacle on right" every single frame, or wrong hint, no visual boxes.
 *
 * Fix: divide cx/cy by frame dimensions before zone/distance tests.
 */

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <sstream>
#include <filesystem>

#include <opencv2/imgproc.hpp>

#include "Config.hpp"
#include "CameraManager.hpp"
#include "ModelRegistry.hpp"
#include "OCREngine.h"
#include "Pipeline.hpp"
#include "ResultsStore.h"
#include "TTSService.hpp"
#include "PythonBridge.hpp"
#include <spdlog/spdlog.h>

namespace SmartGlasses {

using Clock = std::chrono::steady_clock;

static constexpr double SPEAK_COOLDOWN = 4.0;

// â”€â”€â”€ TTS cooldown helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
bool Pipeline::shouldSpeak(const std::string& phrase, float cooldownSec) {
    std::lock_guard<std::mutex> lk(ttsMutex_);
    auto now = std::chrono::steady_clock::now();
    if (firstSpeak_) {
        firstSpeak_    = false;
        lastSpoken_    = phrase;
        lastSpeakTime_ = now;
        return true;
    }
    float elapsed = std::chrono::duration<float>(now - lastSpeakTime_).count();
    if (phrase == lastSpoken_ && elapsed < cooldownSec) return false;
    lastSpoken_    = phrase;
    lastSpeakTime_ = now;
    return true;
}

// â”€â”€â”€ Constructor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Pipeline::Pipeline(CameraManager& camera, ModelRegistry& registry, TTSService& tts)
    : camera_(&camera), registry_(&registry), tts_(&tts) {
    try {
        faceEncoder_ = std::make_unique<hud::FaceEncoder>(Config::modelsDir);
    } catch (const std::exception& e) {
        spdlog::warn("FaceEncoder init failed: {}", e.what());
    }

    try {
        faceStore_ = std::make_unique<hud::IdentityStore>("data/faces.bin");
    } catch (const std::exception& e) {
        spdlog::warn("IdentityStore init failed: {}", e.what());
    }

    try {
        objectEncoder_ = std::make_unique<hud::ObjectEncoder>();
    } catch (const std::exception& e) {
        spdlog::warn("ObjectEncoder init failed: {}", e.what());
    }

    try {
        objectRegistry_ = std::make_unique<hud::ObjectRegistry>("data/objects.json");
    } catch (const std::exception& e) {
        spdlog::warn("ObjectRegistry init failed: {}", e.what());
    }
}

// â”€â”€â”€ Obstacle / Object Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MoveSafeResult Pipeline::runObstacleDetection(cv::Mat& frame, bool speak) {
    MoveSafeResult result;
    result.hint = "Path clear";

    if (!registry_->isYOLOLoaded() || frame.empty())
        return result;

    // Capture (full-resolution) dimensions for normalisation and HUD overlay
    const int captureW = frame.cols;
    const int captureH = frame.rows;
    if (captureW <= 0 || captureH <= 0) return result;

    // Detection resolution from settings (performance knob).
    // 0 or negative â†’ fall back to full capture resolution.
    int detW = (Config::cameraWidth  > 0) ? Config::cameraWidth  : captureW;
    int detH = (Config::cameraHeight > 0) ? Config::cameraHeight : captureH;

    // Create downscaled copy for YOLO if requested.
    cv::Mat detFrame;
    if (detW != captureW || detH != captureH) {
        cv::resize(frame, detFrame, cv::Size(detW, detH), 0, 0, cv::INTER_AREA);
    } else {
        detFrame = frame;
    }

    // Scale factors to map detection-space pixels back to capture-space pixels.
    const float scaleX = static_cast<float>(captureW) / static_cast<float>(detW);
    const float scaleY = static_cast<float>(captureH) / static_cast<float>(detH);

    // Capture dimensions as floats for normalisation.
    const float W = static_cast<float>(captureW);
    const float H = static_cast<float>(captureH);

    std::vector<std::vector<float>> detections;
    std::vector<int>   classIds;
    std::vector<float> confidences;
    registry_->runYOLO(detFrame, detections, classIds, confidences);

    double minDist = 1.0;
    double maxProb = 0.0;

    // For HUD and custom labels (ResultsStore)
    std::vector<DetectionBox> boxesForStore;

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        if (d.size() < 4) continue;
        if (i >= classIds.size() || i >= confidences.size()) continue;

        // Pixel coordinates returned by YOLO are in detection-space.
        const float x1_det = d[0];
        const float y1_det = d[1];
        const float x2_det = d[2];
        const float y2_det = d[3];

        // Map detection-space pixels back to full capture resolution.
        const float x1p = x1_det * scaleX;
        const float y1p = y1_det * scaleY;
        const float x2p = x2_det * scaleX;
        const float y2p = y2_det * scaleY;

        // â”€â”€ Draw rectangles & labels directly on the HUD frame,
        //    mirroring object_detection_verify behaviour â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        {
            cv::Rect bbox(
                static_cast<int>(x1p),
                static_cast<int>(y1p),
                static_cast<int>(x2p - x1p),
                static_cast<int>(y2p - y1p));

            float conf = std::max(0.f, std::min(1.f, confidences[i]));
            int   cls  = classIds[i];

            std::string name = registry_->getYOLOClassName(cls);
            if (name.empty())
                name = "id=" + std::to_string(cls);

            std::string lbl = name + " " + cv::format("%.0f%%", conf * 100.f);

            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 255), 2);
            int baseLine = 0;
            cv::Size sz = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseLine);
            int ty = std::max(bbox.y - 4, sz.height + 4);
            cv::rectangle(frame,
                          cv::Point(bbox.x, ty - sz.height - 4),
                          cv::Point(bbox.x + sz.width + 4, ty + 2),
                          cv::Scalar(0, 255, 255), cv::FILLED);
            cv::putText(frame, lbl,
                        cv::Point(bbox.x + 2, ty - 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 0, 0), 1);
        }

        // â”€â”€ Normalise pixel coords to [0, 1] relative to full capture â”€â”€â”€â”€â”€
        float x1n = x1p / W;
        float y1n = y1p / H;
        float x2n = x2p / W;
        float y2n = y2p / H;

        float cx = (x1n + x2n) * 0.5f;   // normalised centre-x  [0, 1]
        float cy = (y1n + y2n) * 0.5f;   // normalised centre-y  [0, 1]

        // Distance: objects near the bottom of frame are closer
        double dist = 1.0 - static_cast<double>(cy);  // 0 = very close, 1 = far

        const bool isObstacle = isObstacleClass(classIds[i]);

        // Only obstacle-like classes affect MoveSafe metrics and spoken hints
        if (isObstacle) {
            if (dist < minDist)
                minDist = dist;
            if (confidences[i] > maxProb)
                maxProb = confidences[i];

            // Lateral zone (left / centre / right thirds)
            if      (cx < 0.33f) result.leftCount++;
            else if (cx > 0.66f) result.rightCount++;
            else                 result.centerCount++;
        }

        // Object re-identification: try to override label using custom registry.
        std::string customLabel;
        if (objectEncoder_ && objectRegistry_) {
            cv::Rect roi = cv::Rect(
                cv::Point(static_cast<int>(x1p), static_cast<int>(y1p)),
                cv::Point(static_cast<int>(x2p), static_cast<int>(y2p)))
                & cv::Rect(0, 0, captureW, captureH);
            if (roi.width > 8 && roi.height > 8) {
                auto emb = objectEncoder_->Encode(frame(roi).clone());
                auto match = objectRegistry_->FindBest(emb, objectMatchThreshold_);
                if (match.matched)
                    customLabel = match.label;
            }
        }

        // Always expose boxes for HUD overlay:
        // [x1_norm, y1_norm, x2_norm, y2_norm, classId, confidence]
        result.boxes.push_back({
            x1n, y1n, x2n, y2n,
            static_cast<float>(classIds[i]),
            confidences[i]
        });

        DetectionBox db{};
        db.x1 = x1n;
        db.y1 = y1n;
        db.x2 = x2n;
        db.y2 = y2n;
        db.classId = classIds[i];
        db.confidence = confidences[i];
        db.label = customLabel; // empty => HUD falls back to COCO label
        boxesForStore.push_back(std::move(db));
    }

    result.minDistanceNorm = minDist;
    result.maxProbability  = maxProb;

    if (!boxesForStore.empty()) {
        gResultsStore.setBoxes(std::move(boxesForStore));
    }

    // â”€â”€ MoveSafe decision â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (!result.boxes.empty() && minDist <= Config::MOVESAFE_D_THRESHOLD) {
        if (result.centerCount > 0)
            result.hint = "Obstacle ahead - slow down";
        else if (result.leftCount > 0 && result.rightCount == 0)
            result.hint = "Obstacle on left - move right";
        else if (result.rightCount > 0 && result.leftCount == 0)
            result.hint = "Obstacle on right - move left";
        else
            result.hint = "Obstacles on both sides - slow down";
    }

    if (shouldSpeak(result.hint))
        if (speak && Config::TTS_ENABLED) tts_->speak(result.hint);

    return result;
}

// â”€â”€â”€ OCR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
OCRResult Pipeline::runOCR(cv::Mat& frame, bool speak) {
    OCRResult result;
    if (frame.empty()) return result;

    // Write frame to a temporary image and delegate OCR to PythonBridge.
    namespace fs = std::filesystem;
    fs::path tmpPath = fs::temp_directory_path() / "sg_ocr_frame.png";
    if (!cv::imwrite(tmpPath.string(), frame)) {
        return result;
    }

    // Prefer GPU-accelerated ONNX OCR first, fall back to original OCR on empty.
    std::string text = PythonBridge::runOCRonnx(tmpPath.string());
    if (text.empty()) {
        text = PythonBridge::runOCR(tmpPath.string());
    }
    fs::remove(tmpPath);

    text = formatOCRText(text);
    result.text       = text;
    // Python OCR does not return confidence; using presence-only placeholder (0.5 when text present)
    result.confidence = text.empty() ? 0.f : 0.5f;

    if (result.confidence >= Config::OCR_CONFIDENCE_THRESHOLD && !result.text.empty()) {
        if (shouldSpeak(result.text)) {
            if (speak && Config::TTS_ENABLED) tts_->speak(result.text);
        }
    }

    return result;
}

// â”€â”€â”€ Face detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
void Pipeline::runFaceRecognition(cv::Mat& frame, std::vector<cv::Rect>& outFaces, bool speak) {
    static auto s_lastFace = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    auto s_nowFace = std::chrono::steady_clock::now();
    // Run face detection more frequently for a more responsive feel.
    if (std::chrono::duration_cast<std::chrono::milliseconds>(s_nowFace - s_lastFace).count() < 100) return;
    s_lastFace = s_nowFace;
    outFaces.clear();
    if (!registry_->isFaceLoaded() || frame.empty()) return;

    const int captureW = frame.cols;
    const int captureH = frame.rows;
    if (captureW <= 0 || captureH <= 0) return;

    // Detection resolution from settings; 0/negative â†’ use capture resolution.
    int detW = (Config::cameraWidth  > 0) ? Config::cameraWidth  : captureW;
    int detH = (Config::cameraHeight > 0) ? Config::cameraHeight : captureH;

    cv::Mat detFrame;
    if (detW != captureW || detH != captureH) {
        cv::resize(frame, detFrame, cv::Size(detW, detH), 0, 0, cv::INTER_AREA);
    } else {
        detFrame = frame;
    }

    // Scale factors back to capture-space.
    const float scaleX = static_cast<float>(captureW) / static_cast<float>(detW);
    const float scaleY = static_cast<float>(captureH) / static_cast<float>(detH);

        std::vector<cv::Rect> detFaces;
        registry_->runFaceDetection(detFrame, detFaces);

        // Map detection-space face rects back to capture resolution.
        for (const auto& r : detFaces) {
        int x = static_cast<int>(std::round(r.x * scaleX));
        int y = static_cast<int>(std::round(r.y * scaleY));
        int w = static_cast<int>(std::round(r.width * scaleX));
        int h = static_cast<int>(std::round(r.height * scaleY));

        // Clamp to capture frame bounds.
        x = std::max(0, std::min(x, captureW - 1));
        y = std::max(0, std::min(y, captureH - 1));
        if (x + w > captureW) w = captureW - x;
        if (y + h > captureH) h = captureH - y;
        if (w <= 0 || h <= 0) continue;

        outFaces.emplace_back(x, y, w, h);
    }

    // Face recognition: run on detection thread, write names to ResultsStore.
    if (faceEncoder_ && faceStore_ && !outFaces.empty()) {
        std::vector<FaceResult> namedFaces;
        namedFaces.reserve(outFaces.size());
        for (const auto& fr : outFaces) {
            cv::Rect safe = fr & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safe.empty()) {
                namedFaces.push_back(FaceResult{fr, "Unknown", 0.0f});
                continue;
            }
            cv::Mat crop = frame(safe).clone();
            auto emb = faceEncoder_->Encode(crop);
            if (emb.empty()) {
                namedFaces.push_back(FaceResult{fr, "Unknown", 0.0f});
                continue;
            }
            auto match = faceStore_->FindMatch(emb, faceMatchThreshold_);
            std::string name = (match.matched && match.identity)
                ? match.identity->name
                : "Unknown";
            namedFaces.push_back(FaceResult{fr, name, match.similarity});
            if (match.matched && match.identity)
                faceStore_->RecordSighting(match.identity->id);
        }
        gResultsStore.setFaces(std::move(namedFaces));
    }

    if (!speak || outFaces.empty()) return;

    std::string phrase = outFaces.size() == 1
        ? "Person detected"
        : std::to_string(outFaces.size()) + " people detected";

    int fc = static_cast<int>(outFaces.size());
    if (fc != lastFaceCount_) {
        lastFaceCount_ = fc;
        if (speak && Config::TTS_ENABLED && shouldSpeak(phrase)) tts_->speak(phrase);
    }
}

// â”€â”€â”€ Sign-to-text (stub) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
std::string Pipeline::runSignToText(const cv::Mat& /*frame*/) {
    return "";
}

// â”€â”€â”€ OCR tick (live scan + manual capture) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
void Pipeline::tickOCR(const cv::Mat& frame, std::chrono::steady_clock::time_point now) {
    if (frame.empty()) return;

    if (gResultsStore.getOCRCaptureRequested()) {
        gOCREngine.runCaptureScan(frame);
        gResultsStore.setOCRCaptureRequested(false);
        return;
    }

    if (!gResultsStore.getOCRLiveOn()) return;

    float elapsed = std::chrono::duration<float>(now - gResultsStore.getLastOcrScanTime()).count();
    if (elapsed >= 1.0f) {
        gOCREngine.runLiveScan(frame);
        gResultsStore.setLastOcrScanTime(now);
    }
}

// â”€â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
std::string Pipeline::formatOCRText(const std::string& raw) {
    std::string out;
    for (char c : raw)
        if (std::isprint((unsigned char)c)) out += c;
    std::string result;
    bool prevSpace = false;
    for (char c : out) {
        if (c == ' ') { if (!prevSpace) result += c; prevSpace = true; }
        else          { result += c; prevSpace = false; }
    }
    return result;
}

bool Pipeline::isObstacleClass(int classId) {
    // COCO classes relevant to navigation
    static const int obstacles[] = {
        0,   // person
        1,   // bicycle
        2,   // car
        3,   // motorcycle
        5,   // bus
        7,   // truck
        14,  // bench (yes, walkway obstacle)
        24,  // backpack (person carrying)
        56,  // chair
        57,  // couch
        59,  // bed
        60,  // dining table
        62,  // tv / monitor
        63,  // laptop
        66,  // keyboard
        67,  // cell phone
        73,  // book
        -1
    };
    for (int i = 0; obstacles[i] != -1; ++i)
        if (classId == obstacles[i]) return true;
    return false;
}

void Pipeline::captureEnrollmentFrames(int targetIdx, const std::string& label, EnrollmentType type,
                                       std::function<void(float)> progressCb,
                                       std::function<void(bool)> doneCb) {
    (void)label;
    CameraManager* cam = camera_;
    ModelRegistry* reg = registry_;
    if (!cam || !reg) {
        if (doneCb) doneCb(false);
        return;
    }
    std::thread([this, targetIdx, type, progressCb, doneCb, cam, reg, label]() {
        std::vector<cv::Mat> crops;
        crops.reserve(50);
        const int targetSize = (type == EnrollmentType::FACE) ? 128 : 224;
        for (int i = 0; i < 50; ++i) {
            cv::Mat frame;
            cam->readFrame(frame);
            if (frame.empty()) continue;
            cv::Rect cropRoi;
            if (type == EnrollmentType::FACE) {
                std::vector<cv::Rect> faces;
                reg->runFaceDetection(frame, faces);
                if (targetIdx < 0 || targetIdx >= (int)faces.size()) continue;
                cropRoi = faces[targetIdx] & cv::Rect(0, 0, frame.cols, frame.rows);
            } else {
                std::vector<std::vector<float>> dets;
                std::vector<int> cids;
                std::vector<float> confs;
                reg->runYOLO(frame, dets, cids, confs);
                if (targetIdx < 0 || targetIdx >= (int)dets.size() || dets[targetIdx].size() < 4) continue;
                const auto& d = dets[targetIdx];
                cropRoi = cv::Rect(cv::Point((int)d[0], (int)d[1]), cv::Point((int)d[2], (int)d[3]));
                cropRoi &= cv::Rect(0, 0, frame.cols, frame.rows);
            }
            if (cropRoi.width < 8 || cropRoi.height < 8) continue;
            cv::Mat crop = frame(cropRoi).clone();
            cv::Mat resized;
            cv::resize(crop, resized, cv::Size(targetSize, targetSize), 0, 0, cv::INTER_AREA);
            crops.push_back(resized);
            if (progressCb && (i + 1) % 5 == 0)
                progressCb((i + 1) / 50.f);
        }
        {
            std::lock_guard<std::mutex> lock(enrollmentMutex_);
            lastEnrollmentFrames_ = std::move(crops);
        }
        spdlog::debug("Pipeline: enrollment captured {} frames", lastEnrollmentFrames_.size());

        // FACE: build averaged 512-D embedding and persist
        if (type == EnrollmentType::FACE && faceEncoder_ && faceStore_ && !label.empty()) {
            std::vector<cv::Mat> faces;
            {
                std::lock_guard<std::mutex> lk(enrollmentMutex_);
                faces = lastEnrollmentFrames_;
            }
            auto emb = faceEncoder_->BuildIdentityEmbedding(faces);
            if (!emb.empty()) {
                faceStore_->AddIdentity(label, emb, {});
                spdlog::info("Face enrollment stored: '{}'", label);
            }
        }

        // OBJECT: encode each crop and persist all embeddings
        if (type == EnrollmentType::OBJECT && objectEncoder_ && objectRegistry_ && !label.empty()) {
            std::vector<cv::Mat> cropsCopy;
            {
                std::lock_guard<std::mutex> lk(enrollmentMutex_);
                cropsCopy = lastEnrollmentFrames_;
            }
            std::vector<hud::ObjEmbedding> embs;
            for (auto& c : cropsCopy) {
                auto e = objectEncoder_->Encode(c);
                if (!e.empty()) embs.push_back(std::move(e));
            }
            if (!embs.empty()) {
                objectRegistry_->AddObject(label, embs);
                spdlog::info("Object enrollment stored: '{}'", label);
            }
        }

        if (doneCb) doneCb(true);
    }).detach();
}

void Pipeline::getLastEnrollmentFrames(std::vector<cv::Mat>& out) const {
    std::lock_guard<std::mutex> lock(enrollmentMutex_);
    out = lastEnrollmentFrames_;
}

} // namespace SmartGlasses
