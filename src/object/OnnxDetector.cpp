#include "OnnxDetector.hpp"
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <vector>

bool OnnxDetector::enrollObject(const std::vector<cv::Mat>& frames, const std::string& label) {
    (void)label;
    if (frames.empty()) {
        spdlog::error("OnnxDetector::enrollObject: empty frames");
        return false;
    }
    std::vector<float> descriptor;
    const int featSize = 224 * 224 * 3;
    descriptor.resize(static_cast<size_t>(featSize), 0.0f);
    int count = 0;
    for (const auto& f : frames) {
        if (f.empty()) continue;
        cv::Mat resized;
        cv::resize(f, resized, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
        cv::Mat f32;
        resized.convertTo(f32, CV_32FC3);
        if (f32.isContinuous() && f32.total() * 3 >= static_cast<size_t>(featSize)) {
            const float* p = f32.ptr<float>();
            for (int i = 0; i < featSize; ++i)
                descriptor[static_cast<size_t>(i)] += p[i];
            ++count;
        }
    }
    if (count == 0) {
        spdlog::error("OnnxDetector::enrollObject: no valid frames");
        return false;
    }
    float norm = 0.0f;
    for (auto& v : descriptor) {
        v /= static_cast<float>(count);
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f)
        for (auto& v : descriptor) v /= norm;
    spdlog::debug("OnnxDetector: enrollment descriptor computed ({} frames)", count);
    return true;
}
