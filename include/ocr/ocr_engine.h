#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace hud {

struct OcrResult {
    std::string text;
    float       confidence;
    cv::Rect    region;
};

class OcrEngine {
public:
    explicit OcrEngine(const std::string& tessDataPath,
                       const std::string& eastModelPath = "");
    ~OcrEngine();

    // Full frame OCR
    OcrResult RecognizeFrame(const cv::Mat& frame);

    // Crop and recognize a specific region
    OcrResult RecognizeRegion(const cv::Mat& frame, const cv::Rect& region);

    // Find text regions using EAST detector
    std::vector<cv::Rect> DetectTextRegions(const cv::Mat& frame);

    bool HasEAST() const { return hasEAST_; }

private:
    struct TessImpl;
    std::unique_ptr<TessImpl> tess_;
    cv::dnn::Net eastNet_;
    bool hasEAST_ = false;
    cv::Mat PreProcess(const cv::Mat& frame);
};

} // namespace hud
