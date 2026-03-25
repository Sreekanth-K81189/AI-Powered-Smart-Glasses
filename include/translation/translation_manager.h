#pragma once
#include <string>
#include <functional>
#include <unordered_map>

namespace hud {

enum class TaskType {
    NONE,
    SPEECH_TO_TEXT,   TEXT_TO_SPEECH,
    OCR_TO_TEXT,      OCR_TO_SPEECH,
    SIGN_TO_TEXT,     SIGN_TO_SPEECH,
    TEXT_TO_SIGN,     SPEECH_TO_SIGN,
    SCENE_TO_SPEECH,  FACE_TO_SPEECH,
    TRANSLATE_TEXT
};

struct TaskResult {
    TaskType    task;
    std::string inputText;
    std::string outputText;
    std::string outputLang;
    bool        success = false;
};

using TaskCallback = std::function<void(const TaskResult&)>;

class TranslationManager {
public:
    TranslationManager();

    void SetActiveTask(TaskType t);
    TaskType GetActiveTask() const { return activeTask_; }
    void ToggleTask(TaskType t);
    bool IsActive(TaskType t) const { return activeTask_ == t; }

    // LibreTranslate HTTP call
    TaskResult Translate(const std::string& text,
                         const std::string& srcLang,
                         const std::string& tgtLang);

    // Route any input through the active task pipeline
    TaskResult Process(const std::string& input, TaskCallback cb = nullptr);

    void SetLibreTranslateUrl(const std::string& url) { ltUrl_ = url; }
    const std::string& GetTaskName(TaskType t) const;

    // Dedup � don't re-process same text within N seconds
    void SetDedup(float seconds) { dedupSec_ = seconds; }

private:
    TaskType    activeTask_ = TaskType::NONE;
    std::string ltUrl_ = "http://localhost:5000";
    float       dedupSec_ = 3.0f;
    std::string lastInput_;
    float       lastInputTime_ = 0.0f;
    std::unordered_map<TaskType, std::string> taskNames_;
    void InitTaskNames();
};

} // namespace hud
