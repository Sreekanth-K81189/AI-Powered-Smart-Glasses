#include "face/face_detector.h"
#include "face/face_encoder.h"
#include "face/identity_store.h"
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;
namespace hud {

FaceDetector::FaceDetector(const std::string& modelsDir) : modelsDir_(modelsDir) {
    std::string scrfdPath = modelsDir + "/face/scrfd_10g_bnkps.onnx";
    std::string haarPath  = modelsDir + "/face/haarcascade_frontalface_default.xml";

    if (fs::exists(scrfdPath)) {
        LoadSCRFD(scrfdPath);
    } else if (fs::exists(haarPath)) {
        if (haar_.load(haarPath))
            std::cout << "[FaceDetector] Using Haar cascade fallback\n";
        else
            std::cerr << "[FaceDetector] Could not load Haar cascade\n";
    } else {
        std::cerr << "[FaceDetector] No face model found in " << modelsDir << "\n";
    }
}

void FaceDetector::LoadSCRFD(const std::string& path) {
    try {
        scrfdNet_ = cv::dnn::readNetFromONNX(path);
        if (!scrfdNet_.empty()) {
            useSCRFD_ = true;
#ifdef USE_CUDA
            scrfdNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            scrfdNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
            std::cout << "[FaceDetector] SCRFD loaded: " << path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[FaceDetector] SCRFD load failed: " << e.what() << "\n";
    }
}

std::vector<FaceBox> FaceDetector::Detect(const cv::Mat& frame, float confThresh) {
    if (frame.empty()) return {};
    if (useSCRFD_) return RunSCRFD(frame, confThresh);
    return RunHaar(frame);
}

std::vector<FaceBox> FaceDetector::RunSCRFD(const cv::Mat& frame, float thresh) {
    std::vector<FaceBox> results;
    try {
        int netW = 640, netH = 640;
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0/128.0, {netW,netH},
                                              cv::Scalar(127.5,127.5,127.5), true, false);
        scrfdNet_.setInput(blob);
        std::vector<cv::Mat> outs;
        scrfdNet_.forward(outs, scrfdNet_.getUnconnectedOutLayersNames());

        float sx = (float)frame.cols / netW;
        float sy = (float)frame.rows / netH;

        // Parse SCRFD output: [N, 15] = [x1,y1,x2,y2,score, 5*lm_x, 5*lm_y]
        for (auto& out : outs) {
            for (int i = 0; i < out.rows; i++) {
                float score = out.at<float>(i, 4);
                if (score < thresh) continue;
                FaceBox fb;
                float x1 = out.at<float>(i,0)*sx, y1 = out.at<float>(i,1)*sy;
                float x2 = out.at<float>(i,2)*sx, y2 = out.at<float>(i,3)*sy;
                fb.bbox  = cv::Rect((int)x1,(int)y1,(int)(x2-x1),(int)(y2-y1));
                fb.score = score;
                for (int k = 0; k < 5; k++) {
                    fb.landmarks[k] = {out.at<float>(i, 5+k*2)*sx,
                                       out.at<float>(i, 5+k*2+1)*sy};
                }
                results.push_back(fb);
            }
        }
    } catch (...) {}
    return results;
}

std::vector<FaceBox> FaceDetector::RunHaar(const cv::Mat& frame) {
    std::vector<FaceBox> results;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    std::vector<cv::Rect> rects;
    haar_.detectMultiScale(gray, rects, 1.1, 3, 0, {30,30});
    for (auto& r : rects) {
        FaceBox fb;
        fb.bbox  = r;
        fb.score = 1.0f;
        results.push_back(fb);
    }
    return results;
}

bool FaceDetector::enrollFace(const std::vector<cv::Mat>& frames, const std::string& name,
                              FaceEncoder* encoder, IdentityStore* store) {
    if (!encoder || !store || frames.empty()) {
        spdlog::error("FaceDetector::enrollFace: missing encoder, store, or frames");
        return false;
    }
    try {
        Embedding emb = encoder->BuildIdentityEmbedding(frames);
        FaceEncoder::Normalize(emb);
        store->AddIdentity(name, emb, {}, 0.65f);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("FaceDetector::enrollFace: {}", e.what());
        return false;
    }
}

} // namespace hud
