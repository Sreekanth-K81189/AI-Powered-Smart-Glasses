#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    cv::Rect bbox;
    int      classId;
    float    confidence;
};

class OnnxDetector {
public:
    explicit OnnxDetector(const std::string& modelPath);
    std::vector<Detection> detect(cv::Mat& frame);
    bool enrollObject(const std::vector<cv::Mat>& frames, const std::string& label);
    bool isUsingCuda() const { return usingCuda_; }
    static const char* cocoName(int classId);

private:
    Ort::Env                         env_;
    Ort::Session                     session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string              inputName_;
    std::vector<std::string> outputNames_;
    size_t                   numOutputs_ = 0;
    bool                     usingCuda_  = false;

    // Output formats:
    //  RAW    : [1, 84, 8400]   raw YOLOv8 (cx,cy,w,h + 80 class scores)
    //  NMS4   : 4 tensors       num_dets / det_boxes / det_scores / det_classes
    //  NMS6   : [1, N, 6]       single tensor [x1,y1,x2,y2, conf, cls]  <-- fp16_nms
    enum class OutputFormat { RAW, NMS4, NMS6 };
    OutputFormat outputFormat_ = OutputFormat::RAW;

    // NMS4 tensor roles (resolved by name)
    int idxNumDets=0, idxBoxes=1, idxScores=2, idxClasses=3;
    bool boxesNHW_=true;
    int  maxDet_=300;

    // Letterbox state
    float scale_   = 1.f;
    int   padTop_  = 0;
    int   padLeft_ = 0;

    // Whether NMS6 boxes are already in original-image space (auto-detected)
    bool boxesAlreadyMapped_ = false;
    bool spaceChecked_       = false;

    static constexpr int   kInputSize  = 640;
    // Object detection confidence threshold (65%)
    static constexpr float kConfThresh = 0.65f;
    static constexpr float kNmsThresh  = 0.45f;

    void preprocess(const cv::Mat& frame, std::vector<float>& blob);

    std::vector<Detection> postprocessRaw (const cv::Mat& frame,
                                           const float* data,
                                           const std::vector<int64_t>& shape);
    std::vector<Detection> postprocessNMS4(const cv::Mat& frame,
                                           std::vector<Ort::Value>& outputs);
    std::vector<Detection> postprocessNMS6(const cv::Mat& frame,
                                           const float* data,
                                           int numDets);
};
