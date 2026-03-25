#include "PythonBridge.hpp"
#include <cstdio>
#include <string>
#include <thread>
#include <filesystem>
#include <array>

#ifdef _WIN32
#  define POPEN  _popen
#  define PCLOSE _pclose
#else
#  define POPEN  popen
#  define PCLOSE pclose
#endif

namespace SmartGlasses {

static std::string s_baseDir;
static const std::string PYTHON = "py";

static std::string runCmd(const std::string& cmd) {
    std::string result;
    std::array<char, 256> buf{};
    FILE* pipe = POPEN(cmd.c_str(), "r");
    if (!pipe) return "";
    while (fgets(buf.data(), (int)buf.size(), pipe))
        result += buf.data();
    PCLOSE(pipe);
    while (!result.empty() &&
           (result.back()=='\n'||result.back()=='\r'||result.back()==' '))
        result.pop_back();
    return result;
}

void PythonBridge::setBaseDir(const std::string& dir) {
    s_baseDir = dir;
}

void PythonBridge::speakText(const std::string& text) {
    std::string t = text;
    for (auto& c : t) if (c == '"') c = '\'';
    std::thread([t]() {
        namespace fs = std::filesystem;
        std::string script = (fs::path(s_baseDir) / "scripts" / "python" / "tts_service.py").string();
        std::string cmd = PYTHON + " \"" + script + "\" \"" + t + "\" >nul 2>&1";
        runCmd(cmd);
    }).detach();
}

std::string PythonBridge::runSTT(int durationSec) {
    namespace fs = std::filesystem;
    std::string script = (fs::path(s_baseDir) / "scripts" / "python" / "stt_service.py").string();
    std::string cmd = PYTHON + " \"" + script + "\" " + std::to_string(durationSec) + " 2>nul";
    std::string raw = runCmd(cmd);
    auto pos = raw.find("STT_RESULT:");
    if (pos != std::string::npos) return raw.substr(pos + 11);
    if (!raw.empty() && raw.rfind("STT_", 0) != 0) return raw;
    return "";
}

std::string PythonBridge::runSTTFromFile(const std::string& wavPath) {
    namespace fs = std::filesystem;
    std::string script = (fs::path(s_baseDir) / "scripts" / "python" / "stt_service.py").string();
    std::string cmd = PYTHON + " \"" + script + "\" \"" + wavPath + "\" 2>nul";
    std::string raw = runCmd(cmd);
    auto pos = raw.find("STT_RESULT:");
    if (pos != std::string::npos) return raw.substr(pos + 11);
    if (!raw.empty() && raw.rfind("STT_", 0) != 0) return raw;
    return "";
}

std::string PythonBridge::runOCR(const std::string& imagePath) {
    namespace fs = std::filesystem;
    std::string script = (fs::path(s_baseDir) / "scripts" / "python" / "ocr_service.py").string();
    std::string cmd = PYTHON + " \"" + script + "\" \"" + imagePath + "\" 2>nul";
    std::string result = runCmd(cmd);
    if (result == "OCR_EMPTY" || result.empty()) return "";
    return result;
}

std::string PythonBridge::runOCRonnx(const std::string& imagePath) {
    namespace fs = std::filesystem;
    std::string script = (fs::path(s_baseDir) / "scripts" / "python" / "ocr_onnx_service.py").string();
    std::string cmd = PYTHON + " \"" + script + "\" \"" + imagePath + "\" 2>nul";
    std::string raw = runCmd(cmd);

    // Parse: "OCR_RESULT:<text>" → return text only.
    if (raw.rfind("OCR_RESULT:", 0) == 0)
        return raw.substr(11);

    // "OCR_EMPTY" or "OCR_ERROR:..." → treat as no text.
    return "";
}

std::string PythonBridge::runSignOnnx(const std::string& imagePath) {
    namespace fs = std::filesystem;
    std::string script = (fs::path(s_baseDir) / "scripts" / "python" / "sign_onnx_service.py").string();
    std::string cmd = PYTHON + " \"" + script + "\" \"" + imagePath + "\" 2>nul";
    // Caller (SignLanguageTranslator) parses SIGN_RESULT / SIGN_EMPTY.
    return runCmd(cmd);
}

} // namespace SmartGlasses
