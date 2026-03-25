#include "face/face_encoder.h"
#include "utils/image_utils.h"
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <filesystem>

namespace hud {

FaceEncoder::FaceEncoder(const std::string& modelsDir) {
    std::string modelPath = modelsDir + "/face/arcface_r100.onnx";
    if (!std::filesystem::exists(modelPath)) {
        // Face embeddings are optional; keep face detection working even when the ArcFace model isn't present.
        return;
    }
    try {
        // Force CPU execution provider to avoid fragile CUDA DLL loading issues.
        session_ = std::make_unique<OrtSession>(modelPath, false);
        std::cout << "[FaceEncoder] ArcFace loaded\n";
    } catch (const std::exception& e) {
        std::cerr << "[FaceEncoder] " << e.what() << "\n";
    }
}

Embedding FaceEncoder::Encode(const cv::Mat& alignedFace) {
    if (!session_ || alignedFace.empty()) return {};

    cv::Mat face112;
    if (alignedFace.cols != 112 || alignedFace.rows != 112)
        cv::resize(alignedFace, face112, {112,112});
    else
        face112 = alignedFace.clone();

    cv::Mat norm = NormalizeArcFace(face112); // (x-127.5)/127.5

    // HWC -> CHW for ONNX
    std::vector<cv::Mat> chans(3);
    cv::split(norm, chans);
    std::vector<float> inputData;
    inputData.reserve(3*112*112);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < 112; y++)
            for (int x = 0; x < 112; x++)
                inputData.push_back(chans[c].at<float>(y,x));

    std::array<int64_t,4> shape{1,3,112,112};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor  = Ort::Value::CreateTensor<float>(
        memInfo, inputData.data(), inputData.size(),
        shape.data(), shape.size());

    const char* inNames[]  = {"data"};
    const char* outNames[] = {"fc1"};

    try {
        std::vector<Ort::Value> inputVec;
        inputVec.push_back(std::move(tensor));
        auto outputs = session_->Run(
            {inNames, inNames+1},
            inputVec,
            {outNames, outNames+1});

        float* data = outputs[0].GetTensorMutableData<float>();
        Embedding emb(data, data + 512);
        Normalize(emb);
        return emb;
    } catch (const std::exception& e) {
        std::cerr << "[FaceEncoder] Inference error: " << e.what() << "\n";
        return {};
    }
}

Embedding FaceEncoder::EncodeFromFrame(const cv::Mat& frame, const FaceBox& box) {
    cv::Mat aligned = AlignFace(frame, box);
    return Encode(aligned);
}

cv::Mat FaceEncoder::AlignFace(const cv::Mat& frame, const FaceBox& box) {
    // Simple crop + resize (landmark-based alignment if available)
    cv::Rect safe = box.bbox & cv::Rect(0,0,frame.cols,frame.rows);
    if (safe.empty()) return {};
    cv::Mat crop;
    cv::resize(frame(safe), crop, {112,112});
    return crop;
}

Embedding FaceEncoder::BuildIdentityEmbedding(const std::vector<cv::Mat>& faceImages) {
    if (faceImages.empty()) return {};

    std::vector<Embedding> embs;
    std::vector<float>     weights;

    for (auto& img : faceImages) {
        auto e = Encode(img);
        if (e.empty()) continue;
        float q = QualityScore(img);
        embs.push_back(e);
        weights.push_back(q);
    }
    if (embs.empty()) return {};

    // Weighted average
    Embedding result(512, 0.0f);
    float totalW = 0.0f;
    for (size_t i = 0; i < embs.size(); i++) {
        for (int j = 0; j < 512; j++)
            result[j] += embs[i][j] * weights[i];
        totalW += weights[i];
    }
    if (totalW > 0) for (auto& v : result) v /= totalW;
    Normalize(result);
    return result;
}

float FaceEncoder::QualityScore(const cv::Mat& face) {
    // Laplacian variance as sharpness proxy
    cv::Mat gray, lap;
    cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    float sharpness = (float)(sigma[0] * sigma[0]);
    return std::min(1.0f, sharpness / 500.0f);
}

float FaceEncoder::CosineSimilarity(const Embedding& a, const Embedding& b) {
    if (a.size() != b.size()) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); i++) dot += a[i]*b[i];
    return dot; // already normalized
}

float FaceEncoder::L2Distance(const Embedding& a, const Embedding& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); i++) s += (a[i]-b[i])*(a[i]-b[i]);
    return std::sqrt(s);
}

void FaceEncoder::Normalize(Embedding& e) {
    float norm = 0.0f;
    for (auto v : e) norm += v*v;
    norm = std::sqrt(norm);
    if (norm > 1e-10f) for (auto& v : e) v /= norm;
}

std::vector<uint8_t> FaceEncoder::ToBytes(const Embedding& e) {
    std::vector<uint8_t> out(e.size() * sizeof(float));
    std::memcpy(out.data(), e.data(), out.size());
    return out;
}

Embedding FaceEncoder::FromBytes(const std::vector<uint8_t>& b) {
    Embedding e(b.size() / sizeof(float));
    std::memcpy(e.data(), b.data(), b.size());
    return e;
}

// Simple base64 (no external dep needed for small vectors)
static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
std::string FaceEncoder::ToBase64(const Embedding& e) {
    auto bytes = ToBytes(e);
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t n = bytes[i] << 16;
        if (i+1 < bytes.size()) n |= bytes[i+1] << 8;
        if (i+2 < bytes.size()) n |= bytes[i+2];
        out += B64[(n>>18)&63];
        out += B64[(n>>12)&63];
        out += (i+1 < bytes.size()) ? B64[(n>>6)&63] : '=';
        out += (i+2 < bytes.size()) ? B64[n&63]      : '=';
    }
    return out;
}

Embedding FaceEncoder::FromBase64(const std::string& b64) {
    // Decode base64 -> bytes -> embedding
    auto findB64 = [](char c) -> int {
        if (c>='A'&&c<='Z') return c-'A';
        if (c>='a'&&c<='z') return c-'a'+26;
        if (c>='0'&&c<='9') return c-'0'+52;
        if (c=='+') return 62; if (c=='/') return 63; return -1;
    };
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < b64.size(); i += 4) {
        int a=findB64(b64[i]), b=findB64(b64[i+1]);
        int c=(i+2<b64.size()&&b64[i+2]!='=')?findB64(b64[i+2]):-1;
        int d=(i+3<b64.size()&&b64[i+3]!='=')?findB64(b64[i+3]):-1;
        if (a<0||b<0) break;
        bytes.push_back((a<<2)|(b>>4));
        if (c>=0) bytes.push_back(((b&15)<<4)|(c>>2));
        if (d>=0) bytes.push_back(((c&3)<<6)|d);
    }
    return FromBytes(bytes);
}

} // namespace hud

