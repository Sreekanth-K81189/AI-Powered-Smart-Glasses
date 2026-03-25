#pragma once
// SignLanguageDetector.hpp - pure OpenCV hand gesture detection
#ifndef SIGN_LANGUAGE_DETECTOR_HPP
#define SIGN_LANGUAGE_DETECTOR_HPP
#include <opencv2/opencv.hpp>
#include <string>
namespace SmartGlasses {
struct GestureResult {
    int         fingerCount = -1;
    std::string label;
    cv::Rect    handROI;
};
class SignLanguageDetector {
public:
    SignLanguageDetector() = default;
    GestureResult detect(const cv::Mat& frame);
    cv::Mat       drawResult(const cv::Mat& frame, const GestureResult& r) const;
private:
    cv::Mat skinMask(const cv::Mat& ycrcb) const;
    int     countFingers(const cv::Mat& mask, cv::Rect& roi) const;
    static constexpr int MIN_HAND_AREA = 6000;
};
} // namespace SmartGlasses
#endif