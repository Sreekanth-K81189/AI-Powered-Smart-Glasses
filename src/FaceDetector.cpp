#include <cmath>
#include <iostream>

#include "FaceDetector.hpp"

// -------------------------------------------------------
// Constructor — loads ResNet SSD DNN model
// -------------------------------------------------------
FaceDetector::FaceDetector(const std::string& protoPath,
                           const std::string& modelPath,
                           float  confThreshold,
                           int    minFaceSize)
    : confThreshold_(confThreshold), minFaceSize_(minFaceSize)
{
    try {
        net_ = cv::dnn::readNetFromCaffe(protoPath, modelPath);
        if (net_.empty()) {
            std::cerr << "[FaceDetector] ERROR: model is empty!\n";
            return;
        }
        // Use CUDA if available, fallback to CPU
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        loaded_ = true;
        std::cout << "[FaceDetector] ResNet SSD loaded (CUDA)\n";
    } catch (...) {
        try {
            // Fallback to CPU
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            loaded_ = true;
            std::cout << "[FaceDetector] ResNet SSD loaded (CPU fallback)\n";
        } catch (const cv::Exception& e) {
            std::cerr << "[FaceDetector] FATAL: " << e.what() << "\n";
            loaded_ = false;
        }
    }
}

// -------------------------------------------------------
// Low light enhancement via CLAHE
// -------------------------------------------------------
cv::Mat FaceDetector::enhanceLowLight(const cv::Mat& frame) {
    if (cv::mean(frame)[0] > 80.0) return frame;
    cv::Mat lab;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch;
    cv::split(lab, ch);
    cv::createCLAHE(2.0, {8,8})->apply(ch[0], ch[0]);
    cv::merge(ch, lab);
    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}

// -------------------------------------------------------
// Rough pose estimation just from box position/shape
// -------------------------------------------------------
std::string FaceDetector::estimatePoseFromBox(const cv::Rect& box,
                                               const cv::Size& frameSize) {
    float aspect = static_cast<float>(box.width) / static_cast<float>(box.height);
    if (aspect < 0.6f) return "PROFILE";
    if (aspect > 1.3f) return "TILTED";

    float cy = box.y + box.height / 2.0f;
    float relY = cy / frameSize.height;
    if (relY < 0.25f) return "UPWARD";
    if (relY > 0.80f) return "DOWNWARD";
    return "FRONTAL";
}

// -------------------------------------------------------
// Main detection
// -------------------------------------------------------
std::vector<FaceDetector::FaceResult> FaceDetector::detect(const cv::Mat& frame) {
    std::vector<FaceResult> results;
    if (!loaded_ || frame.empty()) return results;

    cv::Mat input = enhanceLowLight(frame);

    // Create blob — ResNet SSD expects 300x300, mean subtraction
    cv::Mat blob = cv::dnn::blobFromImage(
        input, 1.0, {300, 300},
        cv::Scalar(104.0, 177.0, 123.0),
        false, false
    );

    net_.setInput(blob);
    cv::Mat detections = net_.forward();

    // Output shape: [1, 1, N, 7]
    // Each row: [_, _, confidence, x1, y1, x2, y2]
    cv::Mat det = detections.reshape(1, static_cast<int>(detections.total() / 7));

    for (int i = 0; i < det.rows; i++) {
        float conf = det.at<float>(i, 2);
        if (conf < confThreshold_) continue;

        int x1 = static_cast<int>(det.at<float>(i, 3) * frame.cols);
        int y1 = static_cast<int>(det.at<float>(i, 4) * frame.rows);
        int x2 = static_cast<int>(det.at<float>(i, 5) * frame.cols);
        int y2 = static_cast<int>(det.at<float>(i, 6) * frame.rows);

        // Clamp to frame
        x1 = std::max(0, x1); y1 = std::max(0, y1);
        x2 = std::min(frame.cols, x2);
        y2 = std::min(frame.rows, y2);

        int w = x2 - x1;
        int h = y2 - y1;

        // Skip too small
        if (w < minFaceSize_ || h < minFaceSize_) continue;

        // Skip non-square aspect ratios (real faces: 0.4 to 1.8)
        float aspect = static_cast<float>(w) / static_cast<float>(h);
        if (aspect < 0.4f || aspect > 1.8f) continue;

        // Skip faces that are < 4% of frame width (definitely false positives)
        if (w < frame.cols * 0.04f) continue;

        // Skip detections near the very edge of frame (usually reflections/artifacts)
        if (x1 < 5 || y1 < 5 || x2 > frame.cols - 5 || y2 > frame.rows - 5) {
            // Only skip if also small
            if (w < frame.cols * 0.08f) continue;
        }

        FaceResult res;
        res.box        = cv::Rect(x1, y1, w, h);
        res.confidence = conf;
        res.tag        = "UNKNOWN";
        res.poseLabel  = estimatePoseFromBox(res.box, frame.size());

        // SSD has no landmarks — set to box center as placeholder
        cv::Point2f center(x1 + w/2.0f, y1 + h/2.0f);
        for (auto& lm : res.landmarks) lm = center;

        results.push_back(res);
    }
    return results;
}
