// SignLanguageDetector.cpp - skin-colour segmentation + convexity defects
#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "SignLanguageDetector.hpp"
namespace SmartGlasses {
cv::Mat SignLanguageDetector::skinMask(const cv::Mat& ycrcb) const {
    cv::Mat mask;
    cv::inRange(ycrcb, cv::Scalar(0,133,77), cv::Scalar(255,173,127), mask);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  k);
    return mask;
}
int SignLanguageDetector::countFingers(const cv::Mat& mask, cv::Rect& roi) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) { roi = cv::Rect(); return -1; }
    auto it = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b){ return cv::contourArea(a) < cv::contourArea(b); });
    if (cv::contourArea(*it) < MIN_HAND_AREA) { roi = cv::Rect(); return -1; }
    roi = cv::boundingRect(*it);
    std::vector<int> hull;
    cv::convexHull(*it, hull);
    if ((int)hull.size() < 3) return 0;
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(*it, hull, defects);
    int fingers = 0;
    for (const auto& d : defects) {
        if (d[3] / 256.0f < 20.f) continue;
        cv::Point s=(*it)[d[0]], e=(*it)[d[1]], f=(*it)[d[2]];
        double a=cv::norm(s-f), b=cv::norm(e-f), c=cv::norm(s-e);
        if (std::acos((a*a+b*b-c*c)/(2*a*b))*180.0/CV_PI < 90.0) fingers++;
    }
    return std::min(fingers+1, 5);
}
GestureResult SignLanguageDetector::detect(const cv::Mat& frame) {
    GestureResult r;
    if (frame.empty()) return r;
    cv::Mat ycrcb;
    cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);
    r.fingerCount = countFingers(skinMask(ycrcb), r.handROI);
    const char* labels[] = {"Fist","One","Peace/Two","Three","Four","Open/Five"};
    r.label = (r.fingerCount >= 0 && r.fingerCount <= 5)
              ? labels[r.fingerCount] : "No hand";
    return r;
}
cv::Mat SignLanguageDetector::drawResult(const cv::Mat& frame, const GestureResult& r) const {
    cv::Mat out = frame.clone();
    if (r.fingerCount >= 0) {
        cv::rectangle(out, r.handROI, cv::Scalar(0,255,128), 2);
        cv::putText(out, r.label,
            cv::Point(r.handROI.x, r.handROI.y-10),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,128), 2);
    }
    return out;
}
} // namespace SmartGlasses