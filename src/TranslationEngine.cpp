/*
LibreTranslate setup (run separately; this app does not start it):
  pip install libretranslate
  libretranslate --host 127.0.0.1 --port 5000 --load-only en,ja,zh,ko,fr,de,ar
*/

#include <fstream>
#include <mutex>

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"
#include "json.hpp"

#include "Config.hpp"
#include "ResultsStore.h"
#include "TTSQueue.h"
#include "TranslationEngine.h"
#include "TranslationQueue.h"
#include <spdlog/spdlog.h>

using nlohmann::json;

static void logLine(const std::string& s) {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    std::ofstream f("camera_log.txt", std::ios::out | std::ios::app);
    if (!f.is_open()) return;
    f << s << "\n";
}

// Current Config class in this codebase doesn't expose TRANSLATE_DEDUP_MS yet.
// Keep the dedup window aligned with the architecture doc for now.
static constexpr int kTranslateDedupMs = 4000;

TranslationEngine::TranslationEngine() {
    lastSpokenTime_ = std::chrono::steady_clock::now() - std::chrono::hours(24);
}

TranslationEngine::~TranslationEngine() {
    stop();
}

void TranslationEngine::start() {
    if (running_) return;
    running_ = true;

    // Ping LibreTranslate: GET /
    try {
        httplib::Client client("127.0.0.1", 5000);
        client.set_connection_timeout(2);
        client.set_read_timeout(2);
        auto res = client.Get("/");
        if (res) {
            available_ = true;
            logLine("TranslationEngine: LibreTranslate reachable");
        } else {
            available_ = false;
            logLine("TranslationEngine: LibreTranslate not reachable — translation disabled");
        }
    } catch (const std::exception& e) {
        available_ = false;
        logLine(std::string("TranslationEngine: ping exception — ") + e.what());
    }

    thread_ = std::thread([this] { run(); });
}

void TranslationEngine::stop() {
    if (!running_) return;
    running_ = false;
    gTranslationQueue.stop();
    if (thread_.joinable()) thread_.join();
}

bool TranslationEngine::isAvailable() const {
    return available_.load();
}

void TranslationEngine::run() {
    while (running_) {
        TranslationJob job;
        if (!gTranslationQueue.waitAndPop(job)) break;

        if (!available_) {
            if (job.source == TranslationJob::Source::OCR)
                gResultsStore.setOCRTranslated("[Translation unavailable]", true);
            else
                gResultsStore.setSTTTranslated("[Translation unavailable]");
            continue;
        }

        std::string result = translate(job.text, job.sourceLang, job.targetLang);
        if (result.empty()) result = job.text;

        if (job.source == TranslationJob::Source::OCR) {
            gResultsStore.setOCRTranslated(result, true);
        } else {
            gResultsStore.setSTTTranslated(result);
        }

        if (!isDuplicate(result) && SmartGlasses::Config::TTS_ENABLED) {
            gTTSQueue.push(result);
            lastSpoken_ = result;
            lastSpokenTime_ = std::chrono::steady_clock::now();
        }
    }
}

std::string TranslationEngine::translate(const std::string& text,
                                        const std::string& sourceLang,
                                        const std::string& targetLang) {
    if (text.empty() || targetLang.empty()) return "";

    nlohmann::json body;
    body["q"]      = text;
    body["source"] = "auto";
    body["target"] = targetLang;
    body["format"] = "text";

    httplib::Client client("127.0.0.1", 5000);
    client.set_connection_timeout(3);

    auto res = client.Post("/translate", body.dump(), "application/json");

    if (res == nullptr) {
        spdlog::warn("TranslationEngine: no response from LibreTranslate");
        return "";
    }
    if (res->status != 200) {
        spdlog::warn("TranslationEngine: HTTP {} from LibreTranslate", res->status);
        return "";
    }
    try {
        auto j = nlohmann::json::parse(res->body);
        if (!j.contains("translatedText")) {
            spdlog::warn("TranslationEngine: missing translatedText field");
            return "";
        }
        return j.at("translatedText").get<std::string>();
    } catch (const std::exception& e) {
        spdlog::warn("TranslationEngine: JSON parse error: {}", e.what());
        return "";
    }
}

bool TranslationEngine::isDuplicate(const std::string& text) const {
    if (lastSpoken_ != text) return false;
    const auto dt = std::chrono::steady_clock::now() - lastSpokenTime_;
    return dt < std::chrono::milliseconds(kTranslateDedupMs);
}

