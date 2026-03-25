#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "utils/ort_session.h"

namespace hud {

struct Detection {
    cv::Rect    bbox;
    float       confidence;
    int         classId;
    std::string className;
    bool        isCustomClass = false; // from user-added objects
};

struct MoveSafeHint {
    std::string message;   // "Path clear" / "Obstacle ahead - slow down"
    std::string severity;  // "safe" / "caution" / "danger"
};

class ObjectDetector {
public:
    explicit ObjectDetector(const std::string& modelsDir);

    std::vector<Detection> Detect(const cv::Mat& frame,
                                  float confThresh = 0.4f,
                                  float nmsThresh  = 0.45f);

    bool enrollObject(const std::vector<cv::Mat>& frames, const std::string& label);

    MoveSafeHint GetMoveSafeHint(const std::vector<Detection>& dets,
                                  int frameW, int frameH);

    void LoadCustomClass(const std::string& className,
                         const std::vector<float>& protoEmbedding);

    const std::vector<std::string>& GetClassNames() const { return classNames_; }

private:
    std::unique_ptr<OrtSession> session_;
    std::vector<std::string>    classNames_;
    int inputW_ = 640, inputH_ = 640;
    cv::Mat PreProcess(const cv::Mat& frame);
    std::vector<Detection> PostProcess(const std::vector<Ort::Value>& outputs,
                                       int origW, int origH,
                                       float confThresh, float nmsThresh);
};

} // namespace hud
