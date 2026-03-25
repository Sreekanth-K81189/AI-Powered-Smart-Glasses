#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>

namespace hud {

enum class CameraSource { NONE, ESP32, USB };

class CameraManager {
public:
    CameraManager();
    ~CameraManager();

    bool OpenESP32(const std::string& url, int timeoutMs = 5000);
    bool OpenUSB(int index = 0);
    bool AutoOpen(); // tries ESP32 first, falls back to USB 0-2

    bool GetFrame(cv::Mat& out);
    void Release();

    CameraSource GetSource() const { return source_; }
    std::string  GetSourceStr() const;
    bool IsOpen() const;
    int  GetFPS() const { return fps_.load(); }

    void SetESP32Url(const std::string& url) { esp32Url_ = url; }

private:
    cv::VideoCapture cap_;
    CameraSource     source_ = CameraSource::NONE;
    std::string      esp32Url_ = "http://10.112.139.57:81/stream";
    std::atomic<int> fps_{0};
    mutable std::mutex mtx_;
    cv::Mat latestFrame_;
    std::thread captureThread_;
    std::atomic<bool> running_{false};
    void CaptureLoop();
};

} // namespace hud
