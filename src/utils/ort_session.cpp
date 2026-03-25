#include "utils/ort_session.h"
#include <stdexcept>
#include <iostream>

namespace hud {

OrtSession::OrtSession(const std::string& modelPath, bool useCuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "HUD")
{
    opts_.SetIntraOpNumThreads(4);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    if (useCuda) {
        OrtCUDAProviderOptions cudaOpts{};
        cudaOpts.device_id = 0;
        opts_.AppendExecutionProvider_CUDA(cudaOpts);
        std::cout << "[ORT] Using CUDA execution provider\n";
    }
#endif

    std::wstring wpath(modelPath.begin(), modelPath.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opts_);
    std::cout << "[ORT] Loaded model: " << modelPath << "\n";
}

std::vector<Ort::Value> OrtSession::Run(
    const std::vector<const char*>& inputNames,
    const std::vector<Ort::Value>&  inputTensors,
    const std::vector<const char*>& outputNames)
{
    return session_->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        inputTensors.data(),
        inputTensors.size(),
        outputNames.data(),
        outputNames.size());
}

} // namespace hud
