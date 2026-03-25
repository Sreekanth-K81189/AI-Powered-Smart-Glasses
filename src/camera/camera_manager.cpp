#include "camera/camera_manager.h"
#include <iostream>
#include <chrono>

namespace hud {

CameraManager::CameraManager() {}
CameraManager::~CameraManager() { Release(); }

bool CameraManager::OpenESP32(const std::string& url, int timeoutMs) {
    std::lock_guard<std::mutex> lk(mtx_);
    cap_.open(url, cv::CAP_FFMPEG);
    if (!cap_.isOpened()) {
        std::cerr << "[Camera] ESP32 open failed: " << url << "\n";
        return false;
    }
    source_   = CameraSource::ESP32;
    esp32Url_ = url;
    running_  = true;
    captureThread_ = std::thread(&CameraManager::CaptureLoop, this);
    std::cout << "[Camera] ESP32 opened: " << url << "\n";
    return true;
}

bool CameraManager::OpenUSB(int index) {
    std::lock_guard<std::mutex> lk(mtx_);
    cap_.open(index, cv::CAP_DSHOW);
    if (!cap_.isOpened()) cap_.open(index);
    if (!cap_.isOpened()) return false;
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    source_  = CameraSource::USB;
    running_ = true;
    captureThread_ = std::thread(&CameraManager::CaptureLoop, this);
    std::cout << "[Camera] USB camera opened (index " << index << ")\n";
    return true;
}

bool CameraManager::AutoOpen() {
    // Try ESP32 first
    if (OpenESP32(esp32Url_)) return true;
    // Fallback USB 0-2
    for (int i = 0; i <= 2; i++)
        if (OpenUSB(i)) return true;
    source_ = CameraSource::NONE;
    std::cerr << "[Camera] No camera found\n";
    return false;
}

void CameraManager::CaptureLoop() {
    int frameCount = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (running_) {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        { std::lock_guard<std::mutex> lk(mtx_); latestFrame_ = frame; }
        frameCount++;
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= 1) {
            fps_ = frameCount;
            frameCount = 0;
            t0 = std::chrono::steady_clock::now();
        }
    }
}

bool CameraManager::GetFrame(cv::Mat& out) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (latestFrame_.empty()) return false;
    out = latestFrame_.clone();
    return true;
}

void CameraManager::Release() {
    running_ = false;
    if (captureThread_.joinable()) captureThread_.join();
    cap_.release();
    source_ = CameraSource::NONE;
}

bool CameraManager::IsOpen() const { return source_ != CameraSource::NONE; }

std::string CameraManager::GetSourceStr() const {
    switch (source_) {
        case CameraSource::ESP32: return "ESP32";
        case CameraSource::USB:   return "USB";
        default:                  return "NONE";
    }
}

} // namespace hud
