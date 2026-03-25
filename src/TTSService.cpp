#include "TTSService.hpp"
#include "ResultsStore.h"
#include "Config.hpp"
#include "PythonBridge.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <algorithm>

static void TTSLog(const std::string& msg) {
    std::ofstream f("hud_log.txt", std::ios::out | std::ios::app);
    if (f.is_open()) { f << msg << "\n"; f.flush(); }
    std::cout << msg << "\n";
}
#ifdef HAVE_ESPEAK
#include <espeak-ng/speak_lib.h>
#endif
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <sapi.h>
#endif

namespace SmartGlasses {

static std::string nowTime() {
    using namespace std::chrono;
    auto t = system_clock::to_time_t(system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
    return std::string(buf);
}

void TTSService::logBackendToFile(const char* backendName) {
    std::ofstream f("hud_log.txt", std::ios::out | std::ios::app);
    if (f.is_open()) {
        f << "[TTS] Backend: " << backendName << "\n";
        f.flush();
    }
}

#ifdef _WIN32
static std::wstring Utf8ToWide(const std::string& text) {
    if (text.empty()) return std::wstring();
    int wlen = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), (int)text.size(), nullptr, 0);
    if (wlen <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), (int)text.size(), &out[0], wlen);
    return out;
}

static bool SetVoiceToDefaultOutput(ISpVoice* pVoice) {
    ISpObjectTokenCategory* pCat = nullptr;
    IEnumSpObjectTokens*    pEnum = nullptr;
    ISpObjectToken*         pTok  = nullptr;
    if (FAILED(CoCreateInstance(CLSID_SpObjectTokenCategory, nullptr, CLSCTX_ALL,
                               IID_ISpObjectTokenCategory, (void**)&pCat)))
        return false;
    if (FAILED(pCat->SetId(SPCAT_AUDIOOUT, FALSE))) { pCat->Release(); return false; }
    if (FAILED(pCat->EnumTokens(nullptr, nullptr, &pEnum))) { pCat->Release(); return false; }
    ULONG fetched = 0;
    pEnum->Next(1, &pTok, &fetched);
    pEnum->Release();
    pCat->Release();
    if (fetched == 0 || !pTok) return false;
    HRESULT hr = pVoice->SetOutput(pTok, TRUE);
    pTok->Release();
    return SUCCEEDED(hr);
}

// Sanitize for PowerShell: replace every ' with ''
static std::string sanitizeForPowerShell(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 16);
    for (char c : text) {
        if (c == '\'') out += "\'\'";
        else out += c;
    }
    return out;
}

// Tier 2: PowerShell fallback (works on any Windows 10/11)
static void speakPowerShell(const std::string& text) {
    std::string safe = sanitizeForPowerShell(text);
    if (safe.empty()) return;
    std::string cmd = "powershell -Command \"Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Volume = 100; $s.Speak('" + safe + "')\"";
    std::system(cmd.c_str());
}

// Tier 3: SAPI
static bool probeSAPI() {
    ISpVoice* pVoice = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_SpVoice, nullptr, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);
    if (SUCCEEDED(hr) && pVoice) {
        pVoice->Release();
        return true;
    }
    return false;
}

static void speakSAPI(const std::string& text) {
    ISpVoice* pVoice = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_SpVoice, nullptr, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);
    if (FAILED(hr) || !pVoice) return;
    if (!SetVoiceToDefaultOutput(pVoice))
        pVoice->SetOutput(nullptr, TRUE);
    pVoice->SetVolume(100);
    pVoice->SetRate(0);
    std::wstring wtext = Utf8ToWide(text);
    if (!wtext.empty())
        pVoice->Speak(wtext.c_str(), SPF_DEFAULT, nullptr);
    pVoice->Release();
}
#endif

TTSService::TTSService() {
    running_.store(true);
    worker_ = std::thread([this]{ workerLoop(); });
}

TTSService::~TTSService() {
    stop();
}

void TTSService::stop() {
    bool expected = true;
    if (running_.compare_exchange_strong(expected, false)) {
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
}

void TTSService::speak(const std::string& text, bool forceUserAction) {
    if (!Config::TTS_ENABLED && !forceUserAction) return;
    if (text.empty()) return;
    try {
        std::lock_guard<std::mutex> lk(mutex_);
        if (queue_.size() < 3) {
            queue_.push(text);
            cv_.notify_one();
        }
    } catch (...) {}
}

void TTSService::workerLoop() {
#ifdef _WIN32
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
#endif
    activeBackend_ = TTSBackend::TTS_POWERSHELL;
    gResultsStore.setSAPIAvailable(true);
    gResultsStore.setTTSStatus("idle");
    while (running_.load()) {
        std::string text;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]{ return !running_.load() || !queue_.empty(); });
            if (!running_.load()) break;
            text = std::move(queue_.front());
            queue_.pop();
        }
        if (text.empty()) continue;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            log_.push_back({text, nowTime()});
            if (log_.size() > 10) log_.pop_front();
        }

        gResultsStore.setTTSStatus("speaking");
        gResultsStore.setTTSPlaying(true);
        gResultsStore.setTTSSpokenText(text);

#ifdef _WIN32
        if (probeSAPI())
            speakSAPI(text);
        else
            speakPowerShell(text);
#else
        PythonBridge::speakText(text);
#endif

        gResultsStore.setTTSPlaying(false);
        gResultsStore.setTTSStatus("idle");
    }
}

std::deque<LogEntry> TTSService::getLog() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return log_;
}

} // namespace SmartGlasses
