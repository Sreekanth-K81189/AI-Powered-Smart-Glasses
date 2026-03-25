#include "utils/image_utils.h"
#include <stdexcept>

namespace hud {

cv::Mat LoadImage(const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) throw std::runtime_error("Cannot load image: " + path);
    return img;
}

cv::Mat ResizePad(const cv::Mat& img, int w, int h) {
    float scale = std::min((float)w / img.cols, (float)h / img.rows);
    int nw = (int)(img.cols * scale), nh = (int)(img.rows * scale);
    cv::Mat resized;
    cv::resize(img, resized, {nw, nh});
    cv::Mat canvas(h, w, img.type(), cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect((w-nw)/2, (h-nh)/2, nw, nh)));
    return canvas;
}

cv::Mat CropFace(const cv::Mat& img, const cv::Rect& bbox, int size) {
    cv::Rect safe = bbox & cv::Rect(0, 0, img.cols, img.rows);
    if (safe.empty()) return cv::Mat();
    cv::Mat crop;
    cv::resize(img(safe), crop, {size, size});
    return crop;
}

std::vector<float> MatToVector(const cv::Mat& img) {
    cv::Mat f;
    img.convertTo(f, CV_32F);
    return std::vector<float>(f.begin<float>(), f.end<float>());
}

cv::Mat NormalizeArcFace(const cv::Mat& face) {
    // ArcFace: BGR, 112x112, (x-127.5)/127.5
    cv::Mat f;
    face.convertTo(f, CV_32FC3);
    f = (f - 127.5f) / 127.5f;
    return f;
}

void DrawBox(cv::Mat& img, const cv::Rect& r, const std::string& label, cv::Scalar color) {
    cv::rectangle(img, r, color, 2);
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(img, {r.x, r.y - ts.height - 4}, {r.x + ts.width, r.y}, color, -1);
    cv::putText(img, label, {r.x, r.y - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);
}

} // namespace hud
