#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "face/face_detector.h"
#include "object/object_detector.h"
#include "translation/translation_manager.h"

struct GLFWwindow;

namespace hud {

struct AddFaceState {
    char              name[128] = {};
    int               captured  = 0;
    bool              saving    = false;
    std::vector<cv::Mat> images;
    bool              active    = false;   // UI mode toggle
    bool              captureRequested = false;
};

struct AddObjectState {
    char              name[128] = {};
    int               captured  = 0;
    bool              saving    = false;
    std::vector<cv::Mat> crops;
    cv::Rect          box;
    bool              boxValid  = false;
    bool              active    = false;
    bool              captureRequested = false;
};

// Old HUD interface: task-based navigation (Layer 1/5, SELECT TASK menu)
enum class HudView {
    TASK_MENU,      // SELECT TASK with 6 buttons
    DETECTION,      // Layer 1: Object + Face detection
    TRANSLATION,    // Layer 2: Translation Hub
    VOICE_ASSIST,   // Layer 3: Voice Assist
    ADD_DATA,       // Layer 4: Add Face / Add Object
    SETTINGS        // Layer 5: Settings
};

struct HudState {
    HudView currentView = HudView::TASK_MENU;

    // Toggles
    bool detectionActive  = false;
    bool faceActive       = false;
    bool ocrActive        = false;
    bool signActive       = false;
    bool sttActive        = false;
    bool voiceAssistActive= false;
    bool voiceFeedback    = true;
    bool showFPS          = true;

    // Camera
    std::string cameraSource = "NONE";
    std::string esp32Url     = "http://10.112.139.57:81/stream";

    // Detection results (updated each frame)
    std::vector<Detection> detections;
    std::vector<FaceBox>   faces;
    std::string            ocrText;
    float                  ocrConfidence = 0.0f;
    std::string            sttText;
    std::string            statusLine;
    std::string            moveSafeHint;
    int                    fps    = 0;

    // Task
    TaskType               activeTask = TaskType::NONE;
    std::string            taskOutput;

    // Add data panel
    AddFaceState           addFace;
    AddObjectState         addObject;

    // Settings
    float detConfThresh  = 0.4f;
    float faceConfThresh = 0.5f;
    float faceMatchThresh= 0.65f;
    int   voiceInterval  = 4; // seconds between voice assist narration
};

class HudLayer {
public:
    HudLayer();
    ~HudLayer();

    bool Init(int width = 1280, int height = 720, const std::string& title = "Smart Glasses HUD");
    void Render(const cv::Mat& frame, HudState& state);
    bool ShouldClose() const;
    void Shutdown();

    GLFWwindow* GetWindow() const { return window_; }

private:
    GLFWwindow* window_     = nullptr;
    unsigned int videoTex_  = 0;
    int winW_ = 1280, winH_ = 720;

    void UploadFrameToTexture(const cv::Mat& frame);
    void DrawVideoBackground();
    void DrawDetectionOverlays(const HudState& state);
    void DrawHeaderBar(const HudState& state);
    void DrawTaskMenu(HudState& state);
    void DrawMainPanel(const cv::Mat& frame, HudState& state);
    void DrawObjectPanel(HudState& state);
    void DrawFacePanel(HudState& state);
    void DrawOcrPanel(HudState& state);
    void DrawSpeechPanel(HudState& state);
    void DrawTranslationPanel(HudState& state);
    void DrawVoiceAssistPanel(HudState& state);
    void DrawSettingsPanel(HudState& state);
    void DrawStatusBar(const HudState& state);
    void HandleObjectSelection(HudState& state);
};

} // namespace hud
