#include "ocr/ocr_engine.h"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace hud {

struct OcrEngine::TessImpl {
    tesseract::TessBaseAPI api;
};

OcrEngine::OcrEngine(const std::string& tessDataPath, const std::string& eastModelPath)
    : tess_(std::make_unique<TessImpl>())
{
    if (tess_->api.Init(tessDataPath.c_str(), "eng")) {
        throw std::runtime_error("[OCR] Tesseract init failed. Check tessdata path: " + tessDataPath);
    }
    tess_->api.SetPageSegMode(tesseract::PSM_AUTO);
    std::cout << "[OCR] Tesseract ready, tessdata: " << tessDataPath << "\n";

    if (!eastModelPath.empty() && std::filesystem::exists(eastModelPath)) {
        try {
            eastNet_ = cv::dnn::readNet(eastModelPath);
            hasEAST_ = true;
            std::cout << "[OCR] EAST text detector loaded\n";
        } catch (...) {}
    }
}

OcrEngine::~OcrEngine() { tess_->api.End(); }

OcrResult OcrEngine::RecognizeFrame(const cv::Mat& frame) {
    cv::Mat gray;
    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame.clone();

    tess_->api.SetImage(gray.data, gray.cols, gray.rows, 1, gray.step);
    char* outText = tess_->api.GetUTF8Text();
    float conf    = tess_->api.MeanTextConf() / 100.0f;

    OcrResult r;
    r.text       = outText ? outText : "";
    r.confidence = conf;
    r.region     = {0, 0, frame.cols, frame.rows};

    delete[] outText;
    return r;
}

OcrResult OcrEngine::RecognizeRegion(const cv::Mat& frame, const cv::Rect& region) {
    cv::Rect safe = region & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe.empty()) return {};
    return RecognizeFrame(frame(safe));
}

std::vector<cv::Rect> OcrEngine::DetectTextRegions(const cv::Mat& frame) {
    if (!hasEAST_) return {{0, 0, frame.cols, frame.rows}};

    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, {320,320},
                                           cv::Scalar(123.68,116.78,103.94), true, false);
    eastNet_.setInput(blob);
    std::vector<cv::Mat> outs;
    std::vector<cv::String> eastOutNames = {"feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"};
    eastNet_.forward(outs, eastOutNames);

    // Decode EAST output (simplified)
    std::vector<cv::Rect> regions;
    float sx = (float)frame.cols / 320.0f;
    float sy = (float)frame.rows / 320.0f;
    auto& scores  = outs[0];
    auto& geom    = outs[1];
    for (int y=0; y<scores.size[2]; y++) for (int x=0; x<scores.size[3]; x++) {
        if (scores.ptr<float>(0,0,y)[x] < 0.5f) continue;
        float x0 = x*4 - geom.ptr<float>(0,3,y)[x];
        float y0 = y*4 - geom.ptr<float>(0,0,y)[x];
        float x1 = x*4 + geom.ptr<float>(0,1,y)[x];
        float y1 = y*4 + geom.ptr<float>(0,2,y)[x];
        regions.push_back({(int)(x0*sx),(int)(y0*sy),(int)((x1-x0)*sx),(int)((y1-y0)*sy)});
    }
    return regions.empty() ? std::vector<cv::Rect>{{0,0,frame.cols,frame.rows}} : regions;
}

} // namespace hud


