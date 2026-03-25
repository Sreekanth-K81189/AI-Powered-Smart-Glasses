#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#ifdef HAVE_TESSERACT
#include <tesseract/baseapi.h>
#endif

struct ResultsStore;
namespace SmartGlasses { class TTSService; }

// Recognised text block from a single OCR pass (legacy compatibility)
struct OCRResult {
    std::string text;
    float       confidence = 0.0f;
    int         x = 0, y = 0, w = 0, h = 0;
};

class OCREngine {
public:
    OCREngine();
    ~OCREngine();

    // Initialise Tesseract and EAST. Tries east_model_path then "models/frozen_east_text_detection.pb".
    bool init(const std::string& tessDataPath, const std::string& eastModelPath = "");

    void setRegistry(ResultsStore* store);
    void setTTSService(SmartGlasses::TTSService* tts) { tts_ = tts; }

    // ----- Google Lens style API -----

    // Live scan: EAST -> regions -> Tesseract. Runs every ~1s from pipeline. Dedup + 2.5s cooldown, confidence > 60%.
    void runLiveScan(const cv::Mat& frame);

    // Capture scan: full frame or crop, PSM_AUTO, high quality. Saves to .txt with timestamp when text found.
    void runCaptureScan(const cv::Mat& frame);

    void setCropRegion(cv::Rect region);
    void clearCropRegion();

    void setLanguage(const std::string& lang) { language_ = lang; }
    float getConfidence() const { return lastConfidence_; }

    // ----- Legacy API (kept for compatibility) -----
    void submitFrame(const cv::Mat& frame);
    std::vector<OCRResult> getResults();
    bool isEnabled()     const { return m_enabled.load(); }
    void setEnabled(bool v)    { m_enabled.store(v);      }
    bool isInitialised() const { return m_init;           }
    bool start();
    void stop();
    bool isRunning() const { return m_running.load(); }
    std::string statusString() const;
    void shutdown();

private:
    std::vector<cv::RotatedRect> detectTextRegionsEAST(const cv::Mat& frame);
    cv::Mat preprocessCrop(const cv::Mat& crop);
    std::string runTesseractOnImage(const cv::Mat& preprocessed, int pageSegMode);
    std::string cleanOCRText(const std::string& raw);
    float computeSimilarity(const std::string& a, const std::string& b);
    void workerLoop();

#ifdef HAVE_TESSERACT
    tesseract::TessBaseAPI m_tess;
#endif

    ResultsStore*                    m_store   = nullptr;
    SmartGlasses::TTSService*        tts_      = nullptr;
    bool                             m_init    = false;
    std::atomic<bool>                m_enabled { false };
    std::atomic<bool>                m_running { false };

    cv::dnn::Net                     eastNet_;
    bool                              eastLoaded_ = false;
    std::string                       language_  = "eng";
    float                             lastConfidence_ = 0.f;
    cv::Rect                          cropRegion_;
    bool                              useCrop_ = false;

    std::thread                       m_worker;
    std::mutex                        m_queueMtx;
    std::queue<cv::Mat>               m_frameQueue;
    std::condition_variable           m_cv;

    std::mutex                        m_resultMtx;
    std::vector<OCRResult>            m_latestResults;
};

extern OCREngine gOCREngine;
