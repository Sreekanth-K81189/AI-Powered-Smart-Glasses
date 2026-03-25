#include "translation/translation_manager.h"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <iostream>

namespace hud {

TranslationManager::TranslationManager() { InitTaskNames(); }

void TranslationManager::InitTaskNames() {
    taskNames_ = {
        {TaskType::NONE,           "None"},
        {TaskType::SPEECH_TO_TEXT, "Speech ? Text"},
        {TaskType::TEXT_TO_SPEECH, "Text ? Speech"},
        {TaskType::OCR_TO_TEXT,    "OCR ? Text"},
        {TaskType::OCR_TO_SPEECH,  "OCR ? Speech"},
        {TaskType::SIGN_TO_TEXT,   "Sign ? Text"},
        {TaskType::SIGN_TO_SPEECH, "Sign ? Speech"},
        {TaskType::SCENE_TO_SPEECH,"Scene ? Speech"},
        {TaskType::FACE_TO_SPEECH, "Face ? Speech"},
        {TaskType::TRANSLATE_TEXT, "Translate Text"}
    };
}

void TranslationManager::SetActiveTask(TaskType t) { activeTask_ = t; }
void TranslationManager::ToggleTask(TaskType t) {
    activeTask_ = (activeTask_ == t) ? TaskType::NONE : t;
}

TaskResult TranslationManager::Translate(const std::string& text,
                                          const std::string& srcLang,
                                          const std::string& tgtLang) {
    TaskResult r;
    r.task      = TaskType::TRANSLATE_TEXT;
    r.inputText = text;
    try {
        nlohmann::json body;
        body["q"]      = text;
        body["source"] = srcLang;
        body["target"] = tgtLang;
        body["format"] = "text";
        auto resp = cpr::Post(
            cpr::Url{ltUrl_ + "/translate"},
            cpr::Header{{"Content-Type","application/json"}},
            cpr::Body{body.dump()},
            cpr::Timeout{3000});

        if (resp.status_code == 200) {
            auto j       = nlohmann::json::parse(resp.text);
            r.outputText = j["translatedText"];
            r.outputLang = tgtLang;
            r.success    = true;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Translation] Error: " << e.what() << "\n";
        r.outputText = text; // fallback: return original
    }
    return r;
}

TaskResult TranslationManager::Process(const std::string& input, TaskCallback cb) {
    TaskResult r;
    r.task      = activeTask_;
    r.inputText = input;
    r.outputText= input;
    r.success   = true;
    if (cb) cb(r);
    return r;
}

const std::string& TranslationManager::GetTaskName(TaskType t) const {
    static std::string unknown = "Unknown";
    auto it = taskNames_.find(t);
    return (it != taskNames_.end()) ? it->second : unknown;
}

} // namespace hud
