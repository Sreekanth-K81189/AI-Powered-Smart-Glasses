// ============================================================
// CameraManager.cpp  –  FINAL CORRECT VERSION
//
// FIX for C2665 (const mutex):
//   getStatus() is declared const in CameraManager.hpp but
//   std::lock_guard needs a non-const mutex reference.
//   Solution: declare mutex_ as mutable in CameraManager.hpp:
//       mutable std::mutex mutex_;
//   This is the only header change needed.
// ============================================================

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "Config.hpp"
#include "CameraManager.hpp"

using namespace std::chrono_literals;

namespace SmartGlasses {

// ─── File logger ──────────────────────────────────────────────────────────
static std::ofstream g_camLog;
static std::mutex    g_logMx;

static void camLog(const std::string& msg) {
    {
        std::lock_guard<std::mutex> lk(g_logMx);
        if (!g_camLog.is_open())
            g_camLog.open("camera_log.txt", std::ios::out | std::ios::trunc);
        g_camLog << msg << "\n";
        g_camLog.flush();
    }
    std::cout << msg << "\n";
}

// ─── USB open helper (tries one index + backend, waits 5s for frames) ────
static bool tryUSB(cv::VideoCapture& cap, int idx, int backend) {
    cap.release();
    std::string backendName =
        (backend == cv::CAP_DSHOW) ? "DSHOW" :
        (backend == cv::CAP_MSMF)  ? "MSMF"  : "AUTO";
    camLog("[Camera] Trying USB index=" + std::to_string(idx) + " " + backendName);

    if (backend >= 0)
        cap.open(idx, backend);
    else
        cap.open(idx);

    if (!cap.isOpened()) {
        camLog("[Camera] index=" + std::to_string(idx) + " open FAILED");
        return false;
    }

    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    // Match the standalone object_detection_verify capture settings
    // for consistent FPS and GPU-friendly MJPG streaming.
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    // Always request a fixed 1280x720 capture for the base feed.
    int reqW = Config::CAPTURE_WIDTH;
    int reqH = Config::CAPTURE_HEIGHT;
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  reqW);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, reqH);
    cap.set(cv::CAP_PROP_FPS,          30);

    for (int i = 0; i < 50; ++i) {   // up to 5 seconds
        cv::Mat f;
        if (cap.read(f) && !f.empty()) {
            std::ostringstream ok;
            ok << "[Camera] USB index=" << idx << " OK ("
               << (int)cap.get(cv::CAP_PROP_FRAME_WIDTH)  << "x"
               << (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT) << ")";
            camLog(ok.str());
            return true;
        }
        std::this_thread::sleep_for(100ms);
    }
    camLog("[Camera] index=" + std::to_string(idx) + " no frames in 5s");
    cap.release();
    return false;
}

// ─── ESP32 stream helper ──────────────────────────────────────────────────
static bool tryStream(cv::VideoCapture& cap, const std::string& url) {
    cap.release();
    camLog("[Camera] Trying stream: " + url);
    cap.open(url, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        camLog("[Camera] Stream open FAILED");
        return false;
    }
    for (int i = 0; i < 50; ++i) {
        cv::Mat f;
        if (cap.read(f) && !f.empty()) {
            std::ostringstream ok;
            ok << "[Camera] Stream OK (" << f.cols << "x" << f.rows << ")";
            camLog(ok.str());
            return true;
        }
        std::this_thread::sleep_for(100ms);
    }
    camLog("[Camera] Stream opened but no frames in 5s");
    cap.release();
    return false;
}

// ─── Constructor / Destructor ─────────────────────────────────────────────
CameraManager::CameraManager()
    : live_(false)
    , stopCapture_(false)
    , source_(CameraSource::None)
    , status_("[NO CAMERA]")
{}

CameraManager::~CameraManager() { stopFeed(); }

// ─── stopFeed ─────────────────────────────────────────────────────────────
void CameraManager::stopFeed() {
    stopCapture_ = true;
    if (captureThread_.joinable())
        captureThread_.join();
    live_        = false;
    stopCapture_ = false;
}

// ─── readFrame ────────────────────────────────────────────────────────────
void CameraManager::readFrame(cv::Mat& out) {
    std::lock_guard<std::mutex> lk(mutex_);
    out = latestFrame_.clone();
}

bool CameraManager::isLive() const { return live_.load(); }

// ─── getStatus ────────────────────────────────────────────────────────────
// Keep the const here to match the header declaration.
// REQUIRED: add  mutable std::mutex mutex_;  in CameraManager.hpp
// (change  std::mutex mutex_;  to  mutable std::mutex mutex_;)
std::string CameraManager::getStatus() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return status_;
}

// ─── captureLoop ──────────────────────────────────────────────────────────
void CameraManager::captureLoop(cv::VideoCapture cap) {
    int failCount = 0;
    const int MAX_FAIL = 30;

    while (!stopCapture_) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            if (++failCount >= MAX_FAIL) {
                camLog("[Camera] Too many read failures – stopping");
                break;
            }
            std::this_thread::sleep_for(33ms);
            continue;
        }
        failCount = 0;
        // Always store the full-resolution capture frame.
        {
            std::lock_guard<std::mutex> lk(mutex_);
            latestFrame_ = frame.clone();
        }
        std::this_thread::sleep_for(1ms);
    }

    live_ = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        status_ = "[NO SIGNAL]";
        latestFrame_.release();
    }
    camLog("[Camera] Capture loop ended");
}

// ─── startFeed ────────────────────────────────────────────────────────────
bool CameraManager::startFeed() {
    stopFeed();
    stopCapture_ = false;
    live_        = false;
    cameraOpened_.store(false);
    cameraInitDone_.store(false);
    {
        std::lock_guard<std::mutex> lk(mutex_);
        latestFrame_.release();
        status_ = "[INIT CAMERA]";
    }
    camLog("[Camera] startFeed called");

    captureThread_ = std::thread([this]() {
        cv::VideoCapture cap;
        bool opened = false;

        // 1) Try ESP32 stream
        if (!Config::camera_url.empty()) {
            opened = tryStream(cap, Config::camera_url);
            if (opened) {
                source_ = CameraSource::ESP32;
                std::lock_guard<std::mutex> lk(mutex_);
                status_ = "[LIVE] ESP32";
            } else {
                camLog("[Camera] ESP32 failed, trying USB...");
            }
        } else {
            camLog("[Camera] No ESP32 URL, going straight to USB...");
        }

        // 2) USB DirectShow
        if (!opened) {
            camLog("[Camera] Scanning USB (DSHOW) indices 0-5...");
            for (int i = 0; i <= 5 && !opened && !stopCapture_; ++i)
                opened = tryUSB(cap, i, cv::CAP_DSHOW);
            if (opened) {
                source_ = CameraSource::USB;
                std::lock_guard<std::mutex> lk(mutex_);
                status_ = "[LIVE] USB (DSHOW)";
            }
        }

        // 3) USB Media Foundation
        if (!opened) {
            camLog("[Camera] Scanning USB (MSMF) indices 0-5...");
            for (int i = 0; i <= 5 && !opened && !stopCapture_; ++i)
                opened = tryUSB(cap, i, cv::CAP_MSMF);
            if (opened) {
                source_ = CameraSource::USB;
                std::lock_guard<std::mutex> lk(mutex_);
                status_ = "[LIVE] USB (MSMF)";
            }
        }

        // 4) USB auto backend
        if (!opened) {
            camLog("[Camera] Scanning USB (AUTO) indices 0-3...");
            for (int i = 0; i <= 3 && !opened && !stopCapture_; ++i)
                opened = tryUSB(cap, i, -1);
            if (opened) {
                source_ = CameraSource::USB;
                std::lock_guard<std::mutex> lk(mutex_);
                status_ = "[LIVE] USB (AUTO)";
            }
        }

        // 5) Nothing worked
        if (!opened || stopCapture_) {
            camLog("[Camera] !!! No camera found. Check USB connection and drivers !!!");
            source_ = CameraSource::None;
            live_   = false;
            cameraOpened_.store(false);
            {
                std::lock_guard<std::mutex> lk(mutex_);
                status_ = "[NO CAMERA]";
            }
            cameraInitDone_.store(true);
            cameraCV_.notify_one();
            return;
        }

        live_ = true;
        cameraOpened_.store(true);
        camLog("[Camera] Feed started successfully");
        cameraInitDone_.store(true);
        cameraCV_.notify_one();
        captureLoop(std::move(cap));
    });

    {
        std::unique_lock<std::mutex> lk(cameraMtx_);
        // tryUSB() can take up to ~5s to confirm frames; give it enough time to avoid false "not ready" warnings.
        cameraCV_.wait_for(lk, std::chrono::seconds(7), [this]() {
            return cameraInitDone_.load();
        });
    }
    return cameraOpened_.load();
}

} // namespace SmartGlasses
