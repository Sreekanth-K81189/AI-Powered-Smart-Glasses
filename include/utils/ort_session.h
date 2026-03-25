#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace hud {

class OrtSession {
public:
    OrtSession(const std::string& modelPath, bool useCuda = true);
    std::vector<Ort::Value> Run(
        const std::vector<const char*>& inputNames,
        const std::vector<Ort::Value>& inputTensors,
        const std::vector<const char*>& outputNames);
    Ort::Session& GetSession() { return *session_; }

private:
    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
};

} // namespace hud
