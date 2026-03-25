#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <condition_variable>

namespace SmartGlasses {

enum class CameraSource { None, USB, ESP32 };

class CameraManager {
public:
    CameraManager();
    ~CameraManager();
    bool        startFeed();
    void        stopFeed();
    void        readFrame(cv::Mat& out);
    bool        isLive()    const;
    std::string getStatus() const;
private:
    void captureLoop(cv::VideoCapture cap);
    std::thread       captureThread_;
    mutable std::mutex mutex_;
    cv::Mat           latestFrame_;
    std::atomic<bool> live_;
    std::atomic<bool> stopCapture_;
    CameraSource      source_;
    std::string       status_;
    std::atomic<bool> cameraOpened_{false};
    std::atomic<bool> cameraInitDone_{false};
    std::condition_variable cameraCV_;
    std::mutex cameraMtx_;
};

} // namespace SmartGlasses
