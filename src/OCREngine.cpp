#include <chrono>
#include "OCREngine.h"
#include "Config.hpp"
#include "ResultsStore.h"
#include "TTSService.hpp"
#include "PythonBridge.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <filesystem>

#ifdef HAVE_TESSERACT
#include <leptonica/allheaders.h>
#endif

OCREngine gOCREngine;

static void OCRLog(const std::string& msg) {
    std::ofstream f("hud_log.txt", std::ios::out | std::ios::app);
    if (f.is_open()) { f << msg << "\n"; f.flush(); }
    std::cout << msg << "\n";
}

OCREngine::OCREngine() = default;

OCREngine::~OCREngine() {
    shutdown();
}

void OCREngine::shutdown() {
    m_running.store(false);
}

bool OCREngine::init(const std::string& tessDataPath, const std::string& eastModelPath) {
#ifdef HAVE_TESSERACT
    if (m_tess.Init(tessDataPath.c_str(), language_.c_str()) != 0) {
        OCRLog("[OCREngine] Tesseract Init failed: " + tessDataPath);
        return false;
    }
    m_tess.SetVariable("tessedit_char_whitelist", "");
    m_tess.SetVariable("classify_bln_numeric_mode", "0");
    m_init = true;
#else
    (void)tessDataPath;
    OCRLog("[OCREngine] Tesseract not available (HAVE_TESSERACT=0)");
    return false;
#endif

    std::string eastPath = eastModelPath;
    if (eastPath.empty()) {
        eastPath = "models/east_text_detection.pb";
        std::ifstream f(eastPath);
        if (!f.good()) eastPath = "models/frozen_east_text_detection.pb";
    }
    try {
        eastNet_ = cv::dnn::readNet(eastPath);
        if (eastNet_.empty()) {
            OCRLog("[OCREngine] EAST load failed: " + eastPath);
            eastLoaded_ = false;
        } else {
            eastLoaded_ = true;
#if defined(USE_CUDA) || defined(CV_VERSION_EPOCH)
            try {
                eastNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                eastNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            } catch (...) {}
#endif
            OCRLog("[OCREngine] EAST loaded: " + eastPath);
        }
    } catch (const std::exception& e) {
        OCRLog("[OCREngine] EAST exception: " + std::string(e.what()));
        eastLoaded_ = false;
    }
    return m_init;
}

void OCREngine::setRegistry(ResultsStore* store) { m_store = store; }

bool OCREngine::start() {
#ifdef HAVE_TESSERACT
    m_enabled.store(true);
    m_running.store(true);
    return true;
#else
    m_enabled.store(false);
    m_running.store(false);
    return false;
#endif
}

void OCREngine::stop() { m_running.store(false); }

std::string OCREngine::statusString() const {
    if (!m_running.load()) return "disabled";
#ifdef HAVE_TESSERACT
    if (!m_init) return "active (no model)";
    return eastLoaded_ ? "Tesseract+EAST" : "Tesseract";
#else
    return "active (no backend)";
#endif
}

void OCREngine::setCropRegion(cv::Rect region) { cropRegion_ = region; useCrop_ = true; }
void OCREngine::clearCropRegion() { useCrop_ = false; }

// ---------- EAST text region detection ----------
std::vector<cv::RotatedRect> OCREngine::detectTextRegionsEAST(const cv::Mat& frame) {
    std::vector<cv::RotatedRect> out;
    if (!eastLoaded_ || eastNet_.empty() || frame.empty() || frame.cols < 32 || frame.rows < 32)
        return out;

    const int W = 320;
    const int H = 320;
    float scaleW = static_cast<float>(frame.cols) / W;
    float scaleH = static_cast<float>(frame.rows) / H;

    cv::Mat resized, blob;
    cv::resize(frame, resized, cv::Size(W, H));
    cv::dnn::blobFromImage(resized, blob, 1.0, cv::Size(W, H),
                          cv::Scalar(123.68, 116.78, 103.94), true, false);
    eastNet_.setInput(blob);

    std::vector<std::string> layerNames = {
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    };
    std::vector<cv::Mat> outs;
    eastNet_.forward(outs, layerNames);

    cv::Mat scores   = outs[0];
    cv::Mat geometry = outs[1];
    const int rows = scores.size[2];
    const int cols = scores.size[3];
    const float kConfThresh = 0.5f;
    const float kNMSThresh  = 0.4f;

    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;

    for (int y = 0; y < rows; ++y) {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0 = geometry.ptr<float>(0, 0, y);
        const float* x1 = geometry.ptr<float>(0, 1, y);
        const float* x2 = geometry.ptr<float>(0, 2, y);
        const float* x3 = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);

        for (int x = 0; x < cols; ++x) {
            float score = scoresData[x];
            if (score < kConfThresh) continue;

            float offsetX = x * 4.0f;
            float offsetY = y * 4.0f;
            float angle   = anglesData[x];
            float cosA    = std::cos(angle);
            float sinA    = std::sin(angle);
            float h       = x0[x] + x2[x];
            float w       = x1[x] + x3[x];

            cv::Point2f offset(
                offsetX + cosA * x1[x] + sinA * x2[x],
                offsetY - sinA * x1[x] + cosA * x2[x]);
            cv::Point2f p1(-sinA * h + offset.x, -cosA * h + offset.y);
            cv::Point2f p3( cosA * w + offset.x,  sinA * w + offset.y);

            boxes.push_back(cv::RotatedRect(
                0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.f / static_cast<float>(CV_PI)));
            confidences.push_back(score);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, kConfThresh, kNMSThresh, indices);

    for (int idx : indices) {
        cv::RotatedRect box = boxes[idx];
        box.center.x *= scaleW;
        box.center.y *= scaleH;
        box.size.width  *= scaleW;
        box.size.height *= scaleH;

        double area = box.size.width * box.size.height;
        if (area < 400.0) continue;
        double ratio = (std::max(box.size.width, box.size.height) + 1e-6) /
                       (std::min(box.size.width, box.size.height) + 1e-6);
        if (ratio > 20.0) continue;

        out.push_back(box);
    }
    return out;
}

cv::Mat OCREngine::preprocessCrop(const cv::Mat& crop) {
    cv::Mat gray, binary;
    if (crop.channels() == 3)
        cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    else
        gray = crop.clone();

    cv::fastNlMeansDenoising(gray, gray, 10, 7, 21);
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 15, 8);

    double meanVal = cv::mean(binary)[0];
    if (meanVal < 127) cv::bitwise_not(binary, binary);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(binary, binary, kernel, cv::Point(-1, -1), 1);
    return binary;
}

#ifdef HAVE_TESSERACT
std::string OCREngine::runTesseractOnImage(const cv::Mat& preprocessed, int pageSegMode) {
    if (preprocessed.empty() || preprocessed.cols < 8 || preprocessed.rows < 8) return "";
    m_tess.SetPageSegMode(static_cast<tesseract::PageSegMode>(pageSegMode));

    std::vector<uchar> buf;
    cv::imencode(".png", preprocessed, buf);
    if (buf.empty()) return "";
    PIX* pix = pixReadMem(buf.data(), static_cast<size_t>(buf.size()));
    if (!pix) return "";

    m_tess.SetImage(pix);
    m_tess.Recognize(0);
    char* raw = m_tess.GetUTF8Text();
    lastConfidence_ = m_tess.MeanTextConf();
    std::string result(raw ? raw : "");
    if (raw) delete[] raw;
    pixDestroy(&pix);
    return cleanOCRText(result);
}
#else
std::string OCREngine::runTesseractOnImage(const cv::Mat&, int) { return ""; }
#endif

std::string OCREngine::cleanOCRText(const std::string& raw) {
    std::string out = raw;
    out.erase(0, out.find_first_not_of(" \t\n\r"));
    out.erase(out.find_last_not_of(" \t\n\r") + 1);
    static const std::regex multiSpace("  +");
    out = std::regex_replace(out, multiSpace, " ");
    std::istringstream stream(out);
    std::string line, filtered;
    while (std::getline(stream, line)) {
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" "));
        if (trimmed.size() >= 2) filtered += trimmed + "\n";
    }
    if (!filtered.empty() && filtered.back() == '\n') filtered.pop_back();
    return filtered;
}

float OCREngine::computeSimilarity(const std::string& a, const std::string& b) {
    if (a.empty() && b.empty()) return 1.0f;
    const size_t na = a.size(), nb = b.size();
    size_t maxLen = std::max(na, nb);
    if (maxLen == 0) return 1.0f;
    std::vector<size_t> prev(nb + 1), curr(nb + 1);
    for (size_t j = 0; j <= nb; ++j) prev[j] = j;
    for (size_t i = 1; i <= na; ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= nb; ++j) {
            size_t cost = (a[i-1] == b[j-1]) ? 0 : 1;
            curr[j] = std::min({ curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost });
        }
        std::swap(prev, curr);
    }
    size_t d = prev[nb];
    return 1.0f - static_cast<float>(d) / static_cast<float>(maxLen);
}

void OCREngine::runLiveScan(const cv::Mat& frame) {
    // Throttle: only scan every 2 seconds
    static auto s_lastOCR = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    auto s_nowOCR = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(s_nowOCR - s_lastOCR).count() < 2000) return;
    s_lastOCR = s_nowOCR;
    if (!m_init || frame.empty()) return;
#ifdef HAVE_TESSERACT
    std::vector<cv::RotatedRect> boxes = detectTextRegionsEAST(frame);
    if (m_store) {
        gResultsStore.setOCRBoxesRotated(boxes);
    }

    std::string combined;
    float sumConf = 0.f;
    int count = 0;

    for (const auto& box : boxes) {
        cv::RotatedRect padded = box;
        padded.size.width  += 10;
        padded.size.height += 10;

        cv::Mat M = cv::getRotationMatrix2D(padded.center, padded.angle, 1.0);
        cv::Mat rotated, crop;
        cv::warpAffine(frame, rotated, M, frame.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        cv::getRectSubPix(rotated, cv::Size(static_cast<int>(padded.size.width), static_cast<int>(padded.size.height)),
                         padded.center, crop);

        if (crop.rows < 80) {
            float scale = 80.0f / crop.rows;
            cv::resize(crop, crop, cv::Size(), scale, scale, cv::INTER_CUBIC);
        }

        cv::Mat preprocessed = preprocessCrop(crop);
        std::string text = runTesseractOnImage(preprocessed, tesseract::PSM_SINGLE_BLOCK);
        if (!text.empty()) {
            combined += text + " ";
            sumConf += lastConfidence_;
            count++;
        }
    }

    if (!combined.empty()) combined = cleanOCRText(combined);
    float avgConf = (count > 0) ? (sumConf / count) : 0.f;
    if (m_store) {
        gResultsStore.setOCROriginal(combined);
        gResultsStore.setOCRConfidence(avgConf);
    }

    auto now = std::chrono::steady_clock::now();
    std::string lastSpoken = m_store ? gResultsStore.getLastSpokenOcr() : "";
    auto lastTime = m_store ? gResultsStore.getLastOcrSpeakTime() : std::chrono::steady_clock::time_point{};
    float elapsed = std::chrono::duration<float>(now - lastTime).count();

    if (avgConf < 60.0f || combined.empty()) return;
    if (computeSimilarity(combined, lastSpoken) > 0.85f) return;
    if (elapsed < 2.5f) return;

    if (m_store) {
        gResultsStore.setLastSpokenOcr(combined);
        gResultsStore.setLastOcrSpeakTime(now);
    }
    if (tts_ && SmartGlasses::Config::ocrOutputToSpeech && SmartGlasses::Config::TTS_ENABLED)
        tts_->speak(combined, false);
#endif
}

void OCREngine::runCaptureScan(const cv::Mat& frame) {
    if (!m_init || frame.empty()) return;

    cv::Mat input = frame;
    if (useCrop_ && cropRegion_.width > 0 && cropRegion_.height > 0) {
        cv::Rect r = cropRegion_ & cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.area() > 0) input = frame(r);
    }

    namespace fs = std::filesystem;
    fs::path tmpPath = fs::temp_directory_path() / "sg_ocr_capture.png";
    if (!cv::imwrite(tmpPath.string(), input)) {
        OCRLog("[OCREngine] Failed to write temp OCR image");
        return;
    }

    // Prefer GPU-accelerated ONNX OCR first, fall back to original OCR on empty.
    std::string text = SmartGlasses::PythonBridge::runOCRonnx(tmpPath.string());
    if (text.empty()) {
        text = SmartGlasses::PythonBridge::runOCR(tmpPath.string());
    }
    fs::remove(tmpPath);

    if (m_store) {
        gResultsStore.setOCROriginal(text);
        gResultsStore.setOCRConfidence(text.empty() ? 0.f : 1.f);
        gResultsStore.setOCRBoxesRotated(std::vector<cv::RotatedRect>());
    }

    if (!text.empty()) {
        auto t = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count();
        std::string path = "ocr_capture_" + std::to_string(ms) + ".txt";
        std::ofstream f(path);
        if (f.is_open()) {
            f << text;
            f.close();
            OCRLog("[OCREngine] Saved: " + path);
        }
        if (tts_ && SmartGlasses::Config::ocrOutputToSpeech && SmartGlasses::Config::TTS_ENABLED)
            tts_->speak(text, false);
    }
}

void OCREngine::submitFrame(const cv::Mat& frame) {
    if (!m_init || !m_running.load() || frame.empty()) return;
    std::lock_guard<std::mutex> lock(m_queueMtx);
    if (m_frameQueue.size() < 3) m_frameQueue.push(frame.clone());
    m_cv.notify_one();
}

std::vector<OCRResult> OCREngine::getResults() {
    std::lock_guard<std::mutex> lock(m_resultMtx);
    return m_latestResults;
}

void OCREngine::workerLoop() {
    while (m_running.load()) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(m_queueMtx);
            m_cv.wait_for(lock, std::chrono::milliseconds(100), [this] { return !m_frameQueue.empty() || !m_running.load(); });
            if (!m_running.load()) break;
            if (m_frameQueue.empty()) continue;
            frame = m_frameQueue.front();
            m_frameQueue.pop();
        }
        if (frame.empty()) continue;
        runLiveScan(frame);
    }
}
