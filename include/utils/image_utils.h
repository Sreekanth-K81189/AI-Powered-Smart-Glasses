#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace hud {

cv::Mat LoadImage(const std::string& path);
cv::Mat ResizePad(const cv::Mat& img, int w, int h);
cv::Mat CropFace(const cv::Mat& img, const cv::Rect& bbox, int size = 112);
std::vector<float> MatToVector(const cv::Mat& img);
cv::Mat NormalizeArcFace(const cv::Mat& face); // 112x112 BGR -> normalized float
void DrawBox(cv::Mat& img, const cv::Rect& r, const std::string& label, cv::Scalar color);

} // namespace hud
