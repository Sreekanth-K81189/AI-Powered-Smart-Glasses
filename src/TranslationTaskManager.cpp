/*
 * TranslationTaskManager.cpp
 * Routes the active task to the correct Pipeline call.
 * OCR: background thread for EAST region detection when OCR task is active.
 */

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#endif

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <future>

#include "TranslationTaskManager.hpp"
#include "Pipeline.hpp"
#include "TTSService.hpp"
#include "ModelRegistry.hpp"
#include "Config.hpp"
#include "ResultsStore.h"
#include "PythonBridge.hpp"
#include "SignLanguageTranslator.hpp"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

namespace SmartGlasses {

static void Log(const std::string& msg) {
    std::ofstream f("hud_log.txt", std::ios::out | std::ios::app);
    if (f.is_open()) { f << msg << "\n"; f.flush(); }
    std::cout << msg << "\n";
}

static bool isFrameValidForOCR(const cv::Mat& frame) {
    return !frame.empty()
        && frame.cols >= 8
        && frame.rows >= 8
        && frame.data != nullptr
        && (frame.type() == CV_8UC1
            || frame.type() == CV_8UC3
            || frame.type() == CV_8UC4);
}

TranslationTaskManager::TranslationTaskManager(Pipeline& pipeline, TTSService& tts, ModelRegistry* registry)
    : pipeline_(&pipeline), tts_(&tts), registry_(registry) {
    namespace fs = std::filesystem;
    std::string baseDir = fs::path(Config::modelsDir).parent_path().string();

    signLT_ = std::make_unique<SignLanguageTranslator>();
    signLT_->setBaseDir(baseDir);

    if (signLT_->isAvailable()) {
        spdlog::info("[TranslationTaskManager] Sign language translator ready.");
    } else {
        spdlog::warn("[TranslationTaskManager] Sign language translator unavailable.");
    }
}

TranslationTaskManager::~TranslationTaskManager() {
    if (ocrDetectThread_.joinable()) {
        ocrDetectStop_ = true;
        ocrDetectThread_.join();
    }
}

void TranslationTaskManager::setLatestFrameForOCR(const cv::Mat& frame) {
    if (!isFrameValidForOCR(frame)) return;
    std::lock_guard<std::mutex> lock(ocrFrameMutex_);
    latestOcrFrame_ = frame.clone();
}

void TranslationTaskManager::ocrDetectThreadFunc_() {
    if (!registry_) {
        ocrDetectStop_ = true;
        return;
    }
    while (!ocrDetectStop_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        if (ocrDetectStop_.load()) break;
        if (!registry_ || !registry_->isOCRLoaded()) continue;

        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(ocrFrameMutex_);
            if (latestOcrFrame_.empty() ||
                latestOcrFrame_.cols < 8 ||
                latestOcrFrame_.rows < 8)
                continue;
            frame = latestOcrFrame_.clone();
        }
        if (!isFrameValidForOCR(frame)) continue;

        std::vector<OCRRegion> regions;
        bool found = registry_->detectTextRegionsEAST(frame, regions);
        if (found) {
            gResultsStore.setOCRRegions(regions);
            Log("[OCR] EAST found " + std::to_string(regions.size()) + " regions");
        } else {
            gResultsStore.setOCRRegions({});
        }
    }
}

void TranslationTaskManager::setTask(TranslationTask t) {
    // Click same translation task again = turn off (set to NONE)
    static const TranslationTask kTranslationTasks[] = {
        TranslationTask::SPEECH_TO_TEXT, TranslationTask::TEXT_TO_SPEECH,
        TranslationTask::OCR_TO_TEXT, TranslationTask::OCR_TO_SPEECH,
        TranslationTask::SIGN_TO_TEXT, TranslationTask::SIGN_TO_SPEECH,
    };
    bool isTranslation = false;
    for (auto tt : kTranslationTasks)
        if (t == tt) { isTranslation = true; break; }
    if (isTranslation && current_ == t) {
        current_ = TranslationTask::NONE;
        if (ocrDetectThread_.joinable()) {
            ocrDetectStop_ = true;
            ocrDetectThread_.join();
        }
        return;
    }
    TranslationTask prev = current_;
    current_ = t;

    // Google Lens OCR: no background EAST thread — OCR runs only on 2nd tap (runOCROnPendingAndSpeak).
    bool ocrActive = (current_ == TranslationTask::OCR_TO_TEXT || current_ == TranslationTask::OCR_TO_SPEECH);
    bool wasOcrActive = (prev == TranslationTask::OCR_TO_TEXT || prev == TranslationTask::OCR_TO_SPEECH);
    if (!ocrActive && wasOcrActive && ocrDetectThread_.joinable()) {
        ocrDetectStop_ = true;
        ocrDetectThread_.join();
    }
    // Do NOT start ocrDetectThread_ when OCR task selected — avoids FPS drain; use two-tap flow instead.
}

void TranslationTaskManager::runOCRCapture(const cv::Mat& frame) {
    // Throttle: run OCR capture every 6th frame only to maintain FPS
    static int ocrFrameCounter = 0;
    ocrFrameCounter++;
    if (ocrFrameCounter % 6 != 0) return;

    if (!registry_ || !pipeline_) {
        gResultsStore.setOCRStatus("error");
        return;
    }
    if (!isFrameValidForOCR(frame)) return;
    if (current_ != TranslationTask::OCR_TO_TEXT && current_ != TranslationTask::OCR_TO_SPEECH) return;
    gResultsStore.setOCRStatus("processing");
    const bool speak = Config::ocrOutputToSpeech;
    cv::Mat localFrame = frame.clone();

    std::vector<OCRRegion> regions;
    registry_->detectTextRegionsEAST(localFrame, regions);
    if (regions.empty()) {
        gResultsStore.setOCRRegions(regions);
        gResultsStore.setOCROriginal("");
        gResultsStore.setOCRTranslated("", false);
        gResultsStore.setOCRStatus("ready");
        return;
    }

    std::string joined;
    bool first = true;
    for (auto& r : regions) {
        cv::Rect clipped = r.bbox & cv::Rect(0, 0, localFrame.cols, localFrame.rows);
        if (clipped.width < 8 || clipped.height < 8) continue;
        cv::Mat crop = localFrame(clipped).clone();
        std::vector<std::pair<std::string, float>> raw;
        registry_->runOCROnCrop(crop, raw);
        if (raw.empty()) continue;
        auto best = std::max_element(raw.begin(), raw.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; });
        std::string text = Pipeline::formatOCRText(best->first);
        if (text.empty()) continue;
        r.text = text;
        if (!first) joined += " | ";
        joined += text;
        first = false;
    }

    gResultsStore.setOCRRegions(regions);
    gResultsStore.setOCROriginal(joined);
    gResultsStore.setOCRTranslated(joined, !joined.empty());
    gResultsStore.setOCRStatus("ready");
    if (speak && Config::TTS_ENABLED && !joined.empty())
        tts_->speak(joined);
}

void TranslationTaskManager::storeFrameForOCR(const cv::Mat& frame) {
    if (!isFrameValidForOCR(frame)) return;
    std::lock_guard<std::mutex> lock(ocrFrameMutex_);
    pendingOcrFrame_ = frame.clone();
    hasPendingOcrFrame_.store(true);
    gResultsStore.setOCRImageCaptured(true);
    gResultsStore.setOCRStatus("ready");
    Log("[OCR] Image captured — tap again to read aloud");
}

bool TranslationTaskManager::hasPendingOCRFrame() const {
    return hasPendingOcrFrame_.load();
}

void TranslationTaskManager::runOCROnPendingAndSpeak() {
    cv::Mat frame;
    {
        std::lock_guard<std::mutex> lock(ocrFrameMutex_);
        if (pendingOcrFrame_.empty() || !hasPendingOcrFrame_.load()) return;
        frame = pendingOcrFrame_.clone();
        pendingOcrFrame_.release();
        hasPendingOcrFrame_.store(false);
    }
    gResultsStore.setOCRImageCaptured(false);
    gResultsStore.setOCRStatus("processing");
    if (!isFrameValidForOCR(frame)) {
        gResultsStore.setOCRStatus("ready");
        return;
    }
    namespace fs = std::filesystem;
    fs::path tmpPath = fs::temp_directory_path() / "sg_ocr_lens.png";
    if (!cv::imwrite(tmpPath.string(), frame)) {
        gResultsStore.setOCRStatus("ready");
        return;
    }
    // Prefer GPU-accelerated ONNX OCR first, fall back to original OCR on empty.
    std::string text = SmartGlasses::PythonBridge::runOCRonnx(tmpPath.string());
    if (text.empty()) {
        text = SmartGlasses::PythonBridge::runOCR(tmpPath.string());
    }
    fs::remove(tmpPath);
    text = Pipeline::formatOCRText(text);
    gResultsStore.setOCROriginal(text);
    gResultsStore.setOCRConfidence(text.empty() ? 0.f : 100.f);
    gResultsStore.setOCRStatus("ready");
    const bool speak = (current_ == TranslationTask::OCR_TO_SPEECH || Config::ocrOutputToSpeech)
                       && Config::TTS_ENABLED && !text.empty();
    if (speak && tts_)
        tts_->speak(text, true);
}

void TranslationTaskManager::runOCRAtPoint(const cv::Mat& frame, int fx, int fy) {
    if (!registry_ || !pipeline_) {
        gResultsStore.setOCRStatus("error");
        return;
    }
    if (!isFrameValidForOCR(frame) || !registry_->isOCRLoaded()) return;
    if (current_ != TranslationTask::OCR_TO_TEXT && current_ != TranslationTask::OCR_TO_SPEECH) return;
    gResultsStore.setOCRStatus("processing");
    HUDSnapshot snap = gResultsStore.snapshot();
    int foundIndex = -1;
    cv::Rect crop;
    if (fx >= 0 && fy >= 0 && !snap.ocrRegions.empty()) {
        cv::Point pt(fx, fy);
        for (size_t i = 0; i < snap.ocrRegions.size(); ++i) {
            if (snap.ocrRegions[i].bbox.contains(pt)) {
                foundIndex = static_cast<int>(i);
                crop = snap.ocrRegions[i].bbox;
                break;
            }
        }
    }
    if (foundIndex < 0 && fx >= 0 && fy >= 0) {
        int hw = 100, hh = 50;
        int x0 = std::max(0, fx - hw);
        int y0 = std::max(0, fy - hh);
        if (x0 + 2 * hw > frame.cols) x0 = std::max(0, frame.cols - 2 * hw);
        if (y0 + 2 * hh > frame.rows) y0 = std::max(0, frame.rows - 2 * hh);
        crop = cv::Rect(x0, y0, std::min(200, frame.cols - x0), std::min(100, frame.rows - y0));
    } else if (foundIndex < 0) {
        gResultsStore.setOCRStatus("ready");
        return;
    }
    cv::Mat cropMat = frame(crop).clone();
    std::vector<std::pair<std::string, float>> raw;
    registry_->runOCROnCrop(cropMat, raw);
    std::string text;
    if (!raw.empty()) {
        auto best = std::max_element(raw.begin(), raw.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        text = Pipeline::formatOCRText(best->first);
    }
    gResultsStore.setOCROriginal(text);
    gResultsStore.setOCRTranslated(text, !text.empty());
    if (foundIndex >= 0)
        gResultsStore.setSelectedOCRRegion(foundIndex, text);
    gResultsStore.setOCRStatus("ready");
    if (Config::ocrOutputToSpeech && Config::TTS_ENABLED && !text.empty())
        tts_->speak(text);
}

void TranslationTaskManager::runCurrentTask(const cv::Mat& frame, PipelineResult& out) {
    out = PipelineResult{};
    if (frame.empty() || current_ == TranslationTask::NONE) return;

    cv::Mat frameCopy = frame.clone();

    switch (current_) {
        case TranslationTask::SCENE_TO_SPEECH:
            sceneToSpeech_(frameCopy, out);
            break;

        case TranslationTask::FACE_TO_SPEECH:
            faceToSpeech_(frameCopy, out);
            break;

        case TranslationTask::OCR_TO_TEXT:
        case TranslationTask::OCR_TO_SPEECH: {
            // NON-BLOCKING async OCR — render thread never waits for Python
            static std::future<OCRResult> s_ocrFuture;
            static std::atomic<bool> s_ocrRunning{ false };
            static OCRResult s_lastOcrResult{};

            // If a background OCR is done, consume it.
            if (s_ocrRunning.load() && s_ocrFuture.valid() &&
                s_ocrFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    s_lastOcrResult = s_ocrFuture.get();
                } catch (...) {
                    s_lastOcrResult = OCRResult{};
                }
                s_ocrRunning.store(false);
            }

            // Kick a new background OCR if none is running.
            if (!s_ocrRunning.load() && pipeline_ && !frameCopy.empty()) {
                gResultsStore.setOCRStatus("processing");
                s_ocrRunning.store(true);
                // Always run OCR without TTS on background thread; TTS is triggered on main thread below.
                s_ocrFuture = std::async(std::launch::async, [this, frameCopy]() mutable {
                    return pipeline_->runOCR(frameCopy, false);
                });
            }

            // Render thread uses latest available OCR result (cached).
            out.displayText = s_lastOcrResult.text;
            out.confidence  = s_lastOcrResult.confidence;
            gResultsStore.setOCROriginal(s_lastOcrResult.text);
            gResultsStore.setOCRTranslated(s_lastOcrResult.text, !s_lastOcrResult.text.empty());
            gResultsStore.setOCRStatus(s_ocrRunning.load() ? "processing" : "ready");

            // OCR-to-speech is intentionally disabled (project uses OCR -> Text only).
            break;
        }

        case TranslationTask::SIGN_TO_TEXT:
        case TranslationTask::SIGN_TO_SPEECH:
        {
            if (!signLT_ || !signLT_->isAvailable() || frameCopy.empty())
                break;

            // NON-BLOCKING async call — render thread never waits for Python
            SignRecognitionResult result = signLT_->recognizeAsync(frameCopy);

            if (result.detected && !result.word.empty()) {
                gResultsStore.setSignWord(result.word);
                gResultsStore.setSignConfidence(result.confidence);

                if (current_ == TranslationTask::SIGN_TO_SPEECH
                    && Config::TTS_ENABLED
                    && tts_)
                {
                    tts_->speak(result.word, true);
                }
            }
            break;
        }

        case TranslationTask::TEXT_TO_SPEECH:
            out.displayText = "";
            break;
        case TranslationTask::SPEECH_TO_TEXT:
            out.displayText = "";
            break;
        case TranslationTask::TEXT_TO_SIGN:
        case TranslationTask::SPEECH_TO_SIGN:
            out.displayText = "[Not implemented]";
            break;

        default:
            break;
    }
}

void TranslationTaskManager::runDetectionTasks(const cv::Mat& frame,
                                               bool runScene,
                                               bool runFace,
                                               PipelineResult& out) {
    out = PipelineResult{};
    if (frame.empty() || (!runScene && !runFace)) return;

    cv::Mat frameCopy = frame.clone();

    if (runScene) {
        PipelineResult sceneOut;
        sceneToSpeech_(frameCopy, sceneOut);
        out.displayText = sceneOut.displayText;
        out.boxes       = std::move(sceneOut.boxes);
        out.confidence  = sceneOut.confidence;
    }

    if (runFace) {
        PipelineResult faceOut;
        faceToSpeech_(frameCopy, faceOut);
        out.faces = std::move(faceOut.faces);

        if (!runScene) {
            out.displayText = faceOut.displayText;
        } else if (!faceOut.faces.empty()) {
            if (!out.displayText.empty()) out.displayText += " | ";
            out.displayText += faceOut.displayText;
        }
    }
}

void TranslationTaskManager::sceneToSpeech_(const cv::Mat& frame, PipelineResult& out) {
    cv::Mat f = frame.clone();
    auto result    = pipeline_->runObstacleDetection(f);
    out.displayText = result.hint;
    out.boxes       = result.boxes;
    out.confidence  = (float)result.maxProbability;
}

void TranslationTaskManager::faceToSpeech_(const cv::Mat& frame, PipelineResult& out) {
    cv::Mat f = frame.clone();
    pipeline_->runFaceRecognition(f, out.faces);
    out.displayText = out.faces.empty()
        ? "No faces detected"
        : std::to_string(out.faces.size()) + " face(s) detected";
}

void TranslationTaskManager::textToSign_(const std::string& /*text*/, PipelineResult& out) {
    out.displayText = "[ Sign animation placeholder ]";
}

std::string TranslationTaskManager::getTaskLabel(TranslationTask t) const {
    switch (t) {
        case TranslationTask::SPEECH_TO_TEXT:  return "Speech -> Text";
        case TranslationTask::TEXT_TO_SPEECH:  return "Text -> Speech";
        case TranslationTask::OCR_TO_TEXT:     return "OCR -> Text";
        case TranslationTask::OCR_TO_SPEECH:   return "OCR -> Speech";
        case TranslationTask::SIGN_TO_TEXT:    return "Sign -> Text";
        case TranslationTask::SIGN_TO_SPEECH:  return "Sign -> Speech";
        case TranslationTask::TEXT_TO_SIGN:    return "Text -> Sign";
        case TranslationTask::SPEECH_TO_SIGN:  return "Speech -> Sign";
        case TranslationTask::SCENE_TO_SPEECH: return "Scene -> Speech";
        case TranslationTask::FACE_TO_SPEECH:  return "Face -> Speech";
        default: return "Unknown";
    }
}

std::vector<TranslationTask> TranslationTaskManager::getAllTasks() const {
    return {
        TranslationTask::SPEECH_TO_TEXT,
        TranslationTask::TEXT_TO_SPEECH,
        TranslationTask::OCR_TO_TEXT,
        TranslationTask::OCR_TO_SPEECH,
        TranslationTask::SIGN_TO_TEXT,
        TranslationTask::SIGN_TO_SPEECH,
        TranslationTask::TEXT_TO_SIGN,
        TranslationTask::SPEECH_TO_SIGN,
        TranslationTask::SCENE_TO_SPEECH,
        TranslationTask::FACE_TO_SPEECH,
    };
}

} // namespace SmartGlasses
