#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <array>

// -------------------------------------------------------
// FaceDetector — OpenCV DNN + ResNet SSD
// Much better false-positive rejection than YuNet
// Handles frontal, tilted, partial faces
// -------------------------------------------------------
class FaceDetector {
public:
    struct FaceResult {
        cv::Rect                    box;
        float                       confidence;
        std::array<cv::Point2f, 5>  landmarks;   // placeholder (SSD has no landmarks)
        std::string                 poseLabel;   // FRONTAL / TILTED / PROFILE
        std::string                 tag;         // UNKNOWN or name
    };

    FaceDetector(const std::string& protoPath,
                 const std::string& modelPath,
                 float  confThreshold = 0.85f,
                 int    minFaceSize   = 80);

    std::vector<FaceResult> detect(const cv::Mat& frame);

    bool isLoaded() const { return loaded_; }

private:
    cv::dnn::Net net_;
    float        confThreshold_;
    int          minFaceSize_;
    bool         loaded_ = false;

    cv::Mat      enhanceLowLight(const cv::Mat& frame);
    std::string  estimatePoseFromBox(const cv::Rect& box, const cv::Size& frameSize);
};
