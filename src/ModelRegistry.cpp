#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#if HAVE_TESSERACT
#  include <tesseract/baseapi.h>
#  include <tesseract/resultiterator.h>
#endif

#include "Config.hpp"
#include "ModelRegistry.hpp"

namespace SmartGlasses {

static bool isFrameValid(const cv::Mat& frame) {
    return !frame.empty()
        && frame.cols >= 8
        && frame.rows >= 8
        && frame.data != nullptr
        && (frame.type() == CV_8UC1
            || frame.type() == CV_8UC3
            || frame.type() == CV_8UC4);
}

static std::string modelPath(const std::string& rel) {
    // Models are copied next to the executable; paths are relative to CWD.
    return rel;
}

static void Log(const std::string& msg) {
    std::ofstream f("hud_log.txt", std::ios::out | std::ios::app);
    if (f.is_open()) { f << msg << "\n"; f.flush(); }
    std::cout << msg << "\n";
}

static const char* YOLO_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};
static const int YOLO_NUM_CLASSES = 80;

#if HAVE_TESSERACT
void TessDeleter::operator()(void* p) const {
    if (!p) return;
    auto* api = reinterpret_cast<::tesseract::TessBaseAPI*>(p);
    api->End();
    delete api;
}
#else
void TessDeleter::operator()(void* p) const { (void)p; }
#endif

ModelRegistry::~ModelRegistry() = default;

bool ModelRegistry::initialize(const std::string& modelsDir, ProgressCallback progress) {
    int step = 0;
    auto report = [&](const std::string& name) {
        ++step;
        if (progress) progress(step, name);
    };

    if (!loadYOLO(modelsDir)) {
        std::cerr << "ModelRegistry: YOLO load failed (optional if no ONNX)." << std::endl;
    } else {
        report("yolo");
    }
    if (!loadOCR(modelsDir)) {
        std::cerr << "ModelRegistry: OCR load failed." << std::endl;
    } else {
        report("ocr");
    }
    // EAST model lives under models/ocr/
    loadEAST(modelsDir + "/ocr/frozen_east_text_detection.pb");
    if (!eastLoaded_) {
        loadEAST("models/ocr/frozen_east_text_detection.pb");
    }
    if (!loadFace(modelsDir)) {
        std::cerr << "ModelRegistry: Face load failed (using built-in cascade)." << std::endl;
    } else {
        report("face");
    }
    return ocrLoaded_;  // at least OCR required for pipeline
}

bool ModelRegistry::loadYOLO(const std::string& /*modelsDir*/) {
    namespace fs = std::filesystem;
    // Switched to ONNX Runtime â€” supports full YOLOv8x graph unlike cv::dnn.
    // Config::modelsDir is the root "models" folder next to the executable.
    // Project policy: use ONLY yolov8x_fp16.onnx.
    const fs::path onnxPath = fs::path(Config::modelsDir) / "yolo/yolov8x_fp16.onnx";

    if (!fs::exists(onnxPath)) {
        std::cout << "[WARN] YOLO model not found: " << onnxPath.string() << " - skipping." << std::endl;
        yoloLoaded_ = false;
        return false;
    }

    std::cout << "[YOLO] Using yolo/yolov8x_fp16.onnx" << std::endl;

    std::vector<cv::Mat> outs;
    try {
        yoloDetector_ = std::make_unique<OnnxDetector>(onnxPath.string());
        yoloClassNames_.assign(YOLO_NAMES, YOLO_NAMES + YOLO_NUM_CLASSES);
        yoloLoaded_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "YOLO load error (ONNX Runtime): " << e.what() << std::endl;
        yoloLoaded_ = false;
        return false;
    }
}

bool ModelRegistry::loadOCR(const std::string& modelsDir) {
#if !HAVE_TESSERACT
    (void)modelsDir;
    return false;
#else
    (void)modelsDir;
    try {
        auto* api = new ::tesseract::TessBaseAPI();
        std::string tessData = modelPath(Config::tessDataPrefix);
        if (api->Init(tessData.c_str(), "eng", ::tesseract::OEM_LSTM_ONLY) != 0) {
            std::cerr << "[WARN] Tesseract init failed â€” OCR disabled." << std::endl;
    loadWarnings_.push_back("OCR unavailable");
            delete api;
            return false;
        }
        api->SetPageSegMode(::tesseract::PSM_AUTO);
#ifdef _WIN32
        api->SetVariable("debug_file", "NUL");  // suppress "Estimating resolution", "Detected N diacritics" etc.
#else
        api->SetVariable("debug_file", "/dev/null");
#endif
        tess_ = std::unique_ptr<void, TessDeleter>(api, TessDeleter());
        ocrLoaded_ = true;
        return true;
    } catch (...) {
        std::cerr << "[WARN] Tesseract exception â€” OCR disabled." << std::endl;
    loadWarnings_.push_back("OCR unavailable");
        return false;
    }
#endif
}

bool ModelRegistry::loadFace(const std::string& modelsDir) {
    // --- ResNet SSD DNN face detector (preferred) ---
    std::filesystem::path protoPath = std::filesystem::path(Config::modelsDir) / "deploy.prototxt";
    std::filesystem::path modelPath = std::filesystem::path(Config::modelsDir) / "res10_300x300_ssd.caffemodel";

    if (std::filesystem::exists(protoPath) && std::filesystem::exists(modelPath)) {
        try {
            faceDNN_ = cv::dnn::readNetFromCaffe(protoPath.string(), modelPath.string());
            faceDNN_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            faceDNN_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            faceLoaded_ = true;
            std::cout << "[ModelRegistry] ResNet SSD face detector loaded (CPU)\n";
            return true;
        } catch (const cv::Exception& e) {
            std::cerr << "[ModelRegistry] Face DNN load FAILED: " << e.what() << "\n";
        }
    } else {
        std::cerr << "[WARN] ResNet SSD models not found in models folder.\n";
    }

    // --- Fallback: OpenCV Haar cascade so face detection still works ---
    faceLoaded_ = false;
    std::vector<std::string> cascadePaths = {
        (std::filesystem::path(modelsDir) / "face" / "haarcascade_frontalface_default.xml").string(),
        (std::filesystem::path(Config::modelsDir) / "face" / "haarcascade_frontalface_default.xml").string(),
        (std::filesystem::path(Config::modelsDir) / "haarcascade_frontalface_default.xml").string(),
    };
    for (const auto& p : cascadePaths) {
        if (std::filesystem::exists(p) && faceCascade_.load(p)) {
            faceCascadeLoaded_ = true;
            std::cout << "[ModelRegistry] Face detection using Haar cascade: " << p << "\n";
            return true;
        }
    }
    std::cerr << "[ModelRegistry] No face model or cascade found; face detection disabled.\n";
    return false;
}

bool ModelRegistry::isYOLOCuda() const {
    return yoloLoaded_ && yoloDetector_ && yoloDetector_->isUsingCuda();
}

void ModelRegistry::runYOLO(const cv::Mat& frame, std::vector<std::vector<float>>& detections,
                            std::vector<int>& classIds, std::vector<float>& confidences) {
    detections.clear();
    classIds.clear();
    confidences.clear();
    if (!yoloLoaded_ || !yoloDetector_) return;

    cv::Mat frameCopy = frame.clone();
    std::vector<Detection> dets = yoloDetector_->detect(frameCopy);

    for (const auto& d : dets) {
        float x1 = static_cast<float>(d.bbox.x);
        float y1 = static_cast<float>(d.bbox.y);
        float x2 = static_cast<float>(d.bbox.x + d.bbox.width);
        float y2 = static_cast<float>(d.bbox.y + d.bbox.height);

        detections.push_back({x1, y1, x2, y2});
        classIds.push_back(d.classId);
        confidences.push_back(d.confidence);
    }
}

std::string ModelRegistry::getYOLOClassName(int classId) const {
    if (classId < 0 || classId >= static_cast<int>(yoloClassNames_.size()))
        return std::string();
    return yoloClassNames_[classId];
}

void ModelRegistry::runOCR(const cv::Mat& grayOrBgr, std::vector<std::pair<std::string, float>>& results) {
    results.clear();
#if !HAVE_TESSERACT
    (void)grayOrBgr;
    return;
#else
    if (!tess_ || !isFrameValid(grayOrBgr)) return;
    ::tesseract::TessBaseAPI* api = reinterpret_cast<::tesseract::TessBaseAPI*>(tess_.get());
    cv::Mat gray;
    if (grayOrBgr.channels() == 1) {
        gray = grayOrBgr.clone();
    } else {
        cv::cvtColor(grayOrBgr, gray, cv::COLOR_BGR2GRAY);
    }
    if (gray.empty() || gray.cols < 8 || gray.rows < 8) return;

    cv::Mat thresh;
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (thresh.empty()) return;

    cv::Mat contiguous;
    if (!thresh.isContinuous()) {
        contiguous = thresh.clone();
    } else {
        contiguous = thresh;
    }

    api->SetPageSegMode(::tesseract::PSM_AUTO);
    api->SetVariable("preserve_interword_spaces", "1");
    api->SetImage(contiguous.data, contiguous.cols, contiguous.rows, 1, contiguous.step);
    api->SetSourceResolution(70);
    char* text = api->GetUTF8Text();
    if (text) {
        float conf = api->MeanTextConf() / 100.f;
        results.push_back({ std::string(text), conf });
        delete[] text;
    }
    api->GetComponentImages(::tesseract::RIL_TEXTLINE, true, nullptr, nullptr);
#endif
}

void ModelRegistry::loadEAST(const std::string& modelPath) {
    try {
        eastNet_ = cv::dnn::readNet(modelPath);
        if (eastNet_.empty()) {
            Log("[OCR] EAST model failed to load from: " + modelPath);
            eastLoaded_ = false;
            return;
        }

        // Prefer CUDA backend for EAST if OpenCV was built with CUDA support.
        try {
            eastNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            eastNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            Log("[OCR] EAST using CUDA backend");
        } catch (const std::exception& e) {
            Log(std::string("[OCR] EAST CUDA backend unavailable (") + e.what() + ") — falling back to CPU");
            eastNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            eastNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        eastLoaded_ = true;
        Log("[OCR] EAST text detector loaded OK");
    } catch (const std::exception& e) {
        Log("[OCR] EAST load exception: " + std::string(e.what()));
        eastLoaded_ = false;
    }
}

bool ModelRegistry::detectTextRegionsEAST(const cv::Mat& frame, std::vector<OCRRegion>& regions) {
    regions.clear();
    if (!eastLoaded_ || eastNet_.empty()) {
        // Fallback: simple contour detection
        if (!isFrameValid(frame)) return false;
        cv::Mat gray, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (auto& c : contours) {
            cv::Rect r = cv::boundingRect(c);
            if (r.width > 40 && r.height > 12 && r.width > r.height)
                regions.push_back({r, "", 0.5f, false, 0.0f});
        }
        if (regions.size() > 10) regions.resize(10);
        return !regions.empty();
    }
    if (frame.empty() || frame.cols < 32 || frame.rows < 32) return false;

    // EAST requires input size divisible by 32
    int W = ((std::min(frame.cols, 640)) / 32) * 32;
    int H = ((std::min(frame.rows, 640)) / 32) * 32;
    if (W < 32 || H < 32) return false;

    float scaleW = static_cast<float>(frame.cols) / W;
    float scaleH = static_cast<float>(frame.rows) / H;

    std::vector<cv::Mat> outs;
    try {
        cv::Mat blob = cv::dnn::blobFromImage(
            frame, 1.0,
            cv::Size(W, H),
            cv::Scalar(123.68, 116.78, 103.94),
            true, false);

        // Try to use CUDA at inference time; if it fails, fall back to CPU.
        try {
            eastNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            eastNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } catch (const std::exception& e) {
            Log(std::string("[OCR] EAST CUDA backend unavailable at inference (") + e.what() + ") — using CPU");
            eastNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            eastNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        eastNet_.setInput(blob);

        std::vector<std::string> outNames = {
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        };
        eastNet_.forward(outs, outNames);
    } catch (const std::exception& e) {
        Log("[OCR] EAST inference error: " + std::string(e.what()));
        return false;
    }

    cv::Mat scores   = outs[0];
    cv::Mat geometry = outs[1];

    const float kConfThresh = 0.5f;
    const float kNMSThresh  = 0.4f;

    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;

    const int rows = scores.size[2];
    const int cols = scores.size[3];

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
        cv::Rect box = boxes[idx].boundingRect();
        box.x      = static_cast<int>(box.x      * scaleW);
        box.y      = static_cast<int>(box.y      * scaleH);
        box.width  = static_cast<int>(box.width  * scaleW);
        box.height = static_cast<int>(box.height * scaleH);

        box &= cv::Rect(0, 0, frame.cols, frame.rows);

        if (box.width < 20 || box.height < 8) continue;

        OCRRegion r;
        r.bbox          = box;
        r.text          = "";
        r.confidence    = confidences[idx];
        r.selected      = false;
        r.highlightTimer = 0.0f;
        regions.push_back(r);
    }

    std::sort(regions.begin(), regions.end(),
        [](const OCRRegion& a, const OCRRegion& b){
            return a.confidence > b.confidence;
        });
    if (regions.size() > 15) regions.resize(15);

    return !regions.empty();
}

void ModelRegistry::runOCROnCrop(const cv::Mat& crop, std::vector<std::pair<std::string, float>>& results) {
#if !HAVE_TESSERACT
    (void)crop;
    return;
#else
    if (!tess_ || !isFrameValid(crop)) return;
    ::tesseract::TessBaseAPI* api = reinterpret_cast<::tesseract::TessBaseAPI*>(tess_.get());
    cv::Mat gray;
    if (crop.channels() == 1) {
        gray = crop.clone();
    } else {
        cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    }
    if (gray.empty() || gray.cols < 8 || gray.rows < 8) return;

    cv::Mat thresh;
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (thresh.empty()) return;

    cv::Mat contiguous;
    if (!thresh.isContinuous()) {
        contiguous = thresh.clone();
    } else {
        contiguous = thresh;
    }
    api->SetPageSegMode(::tesseract::PSM_SINGLE_BLOCK);
    api->SetVariable("preserve_interword_spaces", "1");
    api->SetImage(contiguous.data, contiguous.cols, contiguous.rows, 1, contiguous.step);
    api->SetSourceResolution(70);
    char* text = api->GetUTF8Text();
    if (text) {
        float conf = api->MeanTextConf() / 100.f;
        results.push_back({ std::string(text), conf });
        delete[] text;
    }
#endif
}

void ModelRegistry::runFaceDetection(const cv::Mat& frame, std::vector<cv::Rect>& faces) {
    faces.clear();
    if ((!faceLoaded_ && !faceCascadeLoaded_) || frame.empty()) return;

    if (faceLoaded_ && !faceDNN_.empty()) {
        // --- ResNet SSD path ---
        cv::Mat input = frame;
        if (cv::mean(frame)[0] < 80.0) {
            cv::Mat lab;
            cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
            std::vector<cv::Mat> ch;
            cv::split(lab, ch);
            cv::createCLAHE(2.0, {8,8})->apply(ch[0], ch[0]);
            cv::merge(ch, lab);
            cv::cvtColor(lab, input, cv::COLOR_Lab2BGR);
        }
        cv::Mat blob = cv::dnn::blobFromImage(
            input, 1.0, {300, 300},
            cv::Scalar(104.0, 177.0, 123.0), false, false);
        faceDNN_.setInput(blob);
        cv::Mat det = faceDNN_.forward();
        cv::Mat detMat = det.reshape(1, static_cast<int>(det.total() / 7));
        for (int i = 0; i < detMat.rows; i++) {
            float conf = detMat.at<float>(i, 2);
            if (conf < 0.85f) continue;
            int x1 = static_cast<int>(detMat.at<float>(i, 3) * frame.cols);
            int y1 = static_cast<int>(detMat.at<float>(i, 4) * frame.rows);
            int x2 = static_cast<int>(detMat.at<float>(i, 5) * frame.cols);
            int y2 = static_cast<int>(detMat.at<float>(i, 6) * frame.rows);
            x1 = std::max(0, x1); y1 = std::max(0, y1);
            x2 = std::min(frame.cols, x2); y2 = std::min(frame.rows, y2);
            int w = x2 - x1, h = y2 - y1;
            if (w < 80 || h < 80) continue;
            float aspect = static_cast<float>(w) / h;
            if (aspect < 0.4f || aspect > 1.8f) continue;
            if (w < frame.cols * 0.05f) continue;
            faces.push_back(cv::Rect(x1, y1, w, h));
        }
        return;
    }

    // --- Haar cascade fallback ---
    if (faceCascadeLoaded_ && !faceCascade_.empty()) {
        cv::Mat gray;
        if (frame.channels() == 3)
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        else
            gray = frame;
        std::vector<cv::Rect> raw;
        faceCascade_.detectMultiScale(gray, raw, 1.1, 5, 0, cv::Size(30, 30));
        for (const auto& r : raw) {
            if (r.width >= 40 && r.height >= 40 && r.width < frame.cols * 0.95f)
                faces.push_back(r);
        }
    }
}

} // namespace SmartGlasses

