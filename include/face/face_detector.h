#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <array>

namespace hud {

class FaceEncoder;
class IdentityStore;

struct FaceBox {
    cv::Rect       bbox;
    float          score;
    std::array<cv::Point2f, 5> landmarks; // 5-point: eyes, nose, mouth corners
    std::string    label;                 // resolved identity name (if known)
};

class FaceDetector {
public:
    explicit FaceDetector(const std::string& modelsDir);
    std::vector<FaceBox> Detect(const cv::Mat& frame, float confThresh = 0.5f);

    // Build embedding from captured face crops and add to identity store.
    bool enrollFace(const std::vector<cv::Mat>& frames, const std::string& name,
                    FaceEncoder* encoder, IdentityStore* store);

private:
    cv::dnn::Net scrfdNet_;       // primary: SCRFD
    cv::CascadeClassifier haar_;  // fallback: Haar cascade
    bool useSCRFD_ = false;
    std::string modelsDir_;
    void LoadSCRFD(const std::string& path);
    std::vector<FaceBox> RunSCRFD(const cv::Mat& frame, float thresh);
    std::vector<FaceBox> RunHaar(const cv::Mat& frame);
};

} // namespace hud
