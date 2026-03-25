#include <opencv2/dnn.hpp>
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include "OnnxDetector.hpp"
#include "ResultsStore.h"

namespace SmartGlasses {

// Tesseract opaque (implementation in .cpp)
struct TessDeleter { void operator()(void*) const; };

// Callback: (step_index, model_name) after each model is loaded.
using ProgressCallback = std::function<void(int, const std::string&)>;

class ModelRegistry {
public:
    const std::vector<std::string>& getWarnings() const { return loadWarnings_; }

    ModelRegistry() = default;
    ~ModelRegistry();

    // Load all models into RAM. Blocks until done. Calls progress(step, name) after each.
    bool initialize(const std::string& modelsDir, ProgressCallback progress = nullptr);

    // ---- Obstacle detection (YOLO) ----
    // Run inference; fill detections as (x1,y1,x2,y2, classId, confidence).
    void runYOLO(const cv::Mat& frame, std::vector<std::vector<float>>& detections,
                 std::vector<int>& classIds, std::vector<float>& confidences);

    // Access to class labels for HUD overlays.
    const std::vector<std::string>& getYOLOClassNames() const { return yoloClassNames_; }
    std::string getYOLOClassName(int classId) const;

    // ---- OCR (Tesseract + EAST text detection) ----
    // Run OCR on image; append (text, confidence) to results. Confidence in [0,1].
    void runOCR(const cv::Mat& grayOrBgr, std::vector<std::pair<std::string, float>>& results);
    // EAST neural network text detection for live overlays (fallback: contour detection if EAST not loaded).
    bool detectTextRegionsEAST(const cv::Mat& frame, std::vector<OCRRegion>& regions);
    // Run OCR on a single crop (e.g. selected region); use PSM_SINGLE_BLOCK. Results appended to results.
    void runOCROnCrop(const cv::Mat& crop, std::vector<std::pair<std::string, float>>& results);

    // ---- Face (OpenCV cascade + optional embedding) ----
    // Detect faces, return rects. Tagging can be added via stored embeddings.
    void runFaceDetection(const cv::Mat& frame, std::vector<cv::Rect>& faces);

    bool isYOLOLoaded() const { return yoloLoaded_; }
      bool isYOLOCuda() const;
    bool isOCRLoaded() const { return ocrLoaded_; }
    bool isFaceLoaded() const { return faceLoaded_ || faceCascadeLoaded_; }

private:
    std::vector<std::string> loadWarnings_;

    bool loadYOLO(const std::string& modelsDir);
    bool loadOCR(const std::string& modelsDir);
    void loadEAST(const std::string& modelPath);
    bool loadFace(const std::string& modelsDir);

    // EAST text detector (replaces MSER)
    cv::dnn::Net eastNet_;
    bool eastLoaded_ = false;

    // YOLO: ONNX Runtime backend (supports full YOLOv8x graph unlike cv::dnn).
    std::unique_ptr<OnnxDetector> yoloDetector_;
    std::vector<std::string> yoloClassNames_;
    bool yoloLoaded_ = false;
    // Ultralytics YOLOv8 defaults: conf=0.25, IoU(NMS)=0.45
    float yoloConfThreshold_ = 0.25f;
    float yoloNmsThreshold_  = 0.45f;
    int yoloInputSize_ = 640;

    // Tesseract (opaque; TessDeleter in .cpp)
    std::unique_ptr<void, TessDeleter> tess_;
    bool ocrLoaded_ = false;

    // Face: ResNet SSD primary; Haar cascade fallback when DNN not available
    cv::CascadeClassifier faceCascade_;
    cv::dnn::Net          faceDNN_;
    bool faceLoaded_ = false;
    bool faceCascadeLoaded_ = false;
};

} // namespace SmartGlasses

