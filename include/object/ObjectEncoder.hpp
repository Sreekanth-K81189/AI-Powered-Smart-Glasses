#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace hud {

using ObjEmbedding = std::vector<float>;

class ObjectEncoder {
public:
    ObjEmbedding Encode(const cv::Mat& crop) const {
        if (crop.empty()) return {};
        cv::Mat r, g, f;
        cv::resize(crop, r, {32, 32});
        cv::cvtColor(r, g, cv::COLOR_BGR2GRAY);
        g.convertTo(f, CV_32F, 1.0f / 255.0f);
        ObjEmbedding emb;
        emb.reserve(32 * 32);
        for (int y = 0; y < f.rows; ++y)
            for (int x = 0; x < f.cols; ++x)
                emb.push_back(f.at<float>(y, x));
        float norm = 0.f;
        for (float v : emb) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-8f) {
            for (auto& v : emb) v /= norm;
        }
        return emb;
    }

    static float Cosine(const ObjEmbedding& a, const ObjEmbedding& b) {
        if (a.size() != b.size() || a.empty()) return 0.f;
        float dot = 0.f;
        for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
        return dot;
    }
};

} // namespace hud

