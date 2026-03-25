#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <array>
#include "face/face_detector.h"
#include "utils/ort_session.h"

namespace hud {

using Embedding = std::vector<float>; // 512-d unit vector

class FaceEncoder {
public:
    explicit FaceEncoder(const std::string& modelsDir);

    // Encode a single aligned face crop -> 512-d embedding
    Embedding Encode(const cv::Mat& alignedFace);

    // Encode from raw frame + detected box (handles alignment internally)
    Embedding EncodeFromFrame(const cv::Mat& frame, const FaceBox& box);

    // Build identity embedding from 5-10 images (quality-weighted average)
    Embedding BuildIdentityEmbedding(const std::vector<cv::Mat>& faceImages);

    // Cosine similarity between two embeddings
    static float CosineSimilarity(const Embedding& a, const Embedding& b);
    static float L2Distance(const Embedding& a, const Embedding& b);

    // Normalize to unit sphere
    static void Normalize(Embedding& e);

    // Serialize/deserialize
    static std::string ToBase64(const Embedding& e);
    static Embedding   FromBase64(const std::string& b64);
    static std::vector<uint8_t> ToBytes(const Embedding& e);
    static Embedding   FromBytes(const std::vector<uint8_t>& bytes);

private:
    std::unique_ptr<OrtSession> session_;
    cv::Mat AlignFace(const cv::Mat& frame, const FaceBox& box);
    float QualityScore(const cv::Mat& face); // blur + pose estimate
};

} // namespace hud
