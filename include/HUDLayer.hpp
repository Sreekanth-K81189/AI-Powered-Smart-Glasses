#pragma once

#include "FaceDetector.hpp"
#include "ResultsStore.h"
#include "TranslationTaskManager.hpp"
#include <string>
#include <vector>
#include <functional>
#include <opencv2/core.hpp>

struct ImGuiContext;
struct ImDrawList;
struct ImVec2;
typedef unsigned int ImU32;
typedef unsigned int GLuint;

namespace hud {
class FaceEncoder;
class IdentityStore;
}

namespace SmartGlasses {

// â”€â”€ Multi-layer task panel states â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
enum class UIPanel {
    Main,            // root: all task categories listed
    ObjectDetection, // (legacy) object detection panel
    FaceDetection,   // (legacy) face detection panel
    OCR,             // sub: OCR->Text / OCR->Speech / Back
    Sign,            // sub: Sign->Text / Sign->Speech / Back
    Speech,          // sub: Speech->Text / Text->Speech / Back
    Translation,     // sub: Text->Sign / Speech->Sign / Back
    VoiceAssist,     // sub: Start / Stop / Back
    AddData,         // sub: Add Face / Add Object / Back
    Settings         // sub: HUD / model settings
};

// Pipeline mode for main loop (Navigation / OCR / Face / Sign)
enum class HUDMode { Navigation, OCR, Speech, Sign, Face };

// Action produced by the HUD that main.cpp should handle
enum class HUDActionType {
    None,
    SetTask,            // task index is stored in pendingTaskIdx_
    ActivateDetection,
    DeactivateDetection,
    StartFaceDetection,
    StopFaceDetection,
    AddFace,            // caller should capture a face frame and save it
    AddObject,          // caller should capture an object sample
    StartVoiceAssist,
    StopVoiceAssist,
    StartSTT,
    StopSTT,
    SpeakText,
    CaptureOCR,
    SetOCROutputText,
    SetOCROutputSpeech,
    SetSignOutputText,
    SetSignOutputSpeech,
    SetVoiceFeedback
};

struct HUDAction {
    HUDActionType type = HUDActionType::None;
    int taskIdx = -1;   // TranslationTask cast index when type==SetTask
    std::string text;   // payload for actions like SpeakText
    int ocrClickX = -1; // frame X when type==CaptureOCR and user clicked on video (else -1 = full-frame)
    int ocrClickY = -1;
};

// â”€â”€ Unified HUD layer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class ModelRegistry;
class HUDLayer {
public:
    void setRegistry(ModelRegistry* r);
    void setVoiceFeedback(bool v) { voiceFeedback_ = v; }
    bool isVoiceFeedbackEnabled() const { return voiceFeedback_; }
    // Optional HUD-owned status line override (used by face enrollment)
    void setStatusLine(const std::string& s) { statusOverride_ = s; }
    HUDLayer();
    ~HUDLayer();

    // Main render call. Call once per frame.
    // fps            â€“ current render fps (displayed in header bar)
    // taskMgr        â€“ pointer to task manager; may be nullptr before init
    void draw(float displayWidth, float displayHeight,
              const cv::Mat& frame,
              const std::string& statusLine,
              const std::string& hintLine,
              float confidencePercent,
              const std::vector<std::vector<float>>& detectionBoxes,
              const std::vector<cv::Rect>& faceRects,
              float fps = 0.f,
              TranslationTaskManager* taskMgr = nullptr);

    // Upload BGR frame â†’ OpenGL texture (BGRâ†’RGB conversion applied).
    GLuint uploadFrame(const cv::Mat& frame);

    void setFontScale(float scale) { fontScale_ = scale; }

    // Poll pending action produced by user interaction this frame.
    // Returns None if no action, then resets to None.
    HUDAction pollAction();

    // Query live flags (set by the HUD panel buttons)
    bool isDetectionActive()     const { return detectionActive_; }
    bool isFaceDetectionActive()  const { return faceDetectionActive_; }
    bool isVoiceAssistActive()    const { return voiceAssistActive_; }

    // Face enrollment state machine for Add Face
    enum class EnrollState { IDLE, WAITING_NAME, CAPTURING, ENCODING, DONE };
    EnrollState getEnrollState() const { return enrollState_; }
    const std::string& getEnrolledName() const { return enrolledName_; }
    size_t getCapturedFaceCount() const { return capturedFaces_.size(); }

private:
    ModelRegistry* registry_     = nullptr;
    bool           voiceFeedback_ = false;
    void handleInput();
    void drawLayerIndicator(float displayW);
    // â”€â”€ Video background â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    void drawVideoBackground(float videoW, float h, GLuint texID);

    // â”€â”€ Overlay (boxes, faces) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    void drawOverlay(float videoW, float h,
                     const std::vector<std::vector<float>>& boxes,
                     const std::vector<cv::Rect>& faces);

    // â”€â”€ Header bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    void drawHeader(float w, float h,
                    const std::string& statusLine,
                    const std::string& hintLine,
                    float confidencePercent,
                    float fps);

    // â”€â”€ Right task panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    void drawTaskPanel(float panelX, float panelW, float h,
                       TranslationTaskManager* taskMgr);

    // Sub-panel drawers
    void drawPanel_Main(TranslationTaskManager* taskMgr);
    void drawPanel_ObjectDetection(TranslationTaskManager* taskMgr);
    void drawPanel_FaceDetection(TranslationTaskManager* taskMgr);
    void drawPanel_AddData(TranslationTaskManager* taskMgr);
    void drawPanel_Settings(TranslationTaskManager* taskMgr);
    void drawPanel_OCR(TranslationTaskManager* taskMgr);
    void drawPanel_Sign(TranslationTaskManager* taskMgr);
    void drawPanel_Speech(TranslationTaskManager* taskMgr);
    void drawPanel_Translation(TranslationTaskManager* taskMgr);
    void drawPanel_VoiceAssist(TranslationTaskManager* taskMgr);

    // Helper: full-width styled button; returns true if clicked
    bool panelButton(const char* label, bool active = false);
    // Helper: active = task is on, selected = keyboard focus (two distinct highlights)
    bool panelButtonEx(const char* label, bool isActive, bool isSelected);

    float    fontScale_   = 1.f;
    GLuint   cameraTexID_ = 0;
    int      lastWidth_   = 0;
    int      lastHeight_  = 0;
    float    lastFps_     = 0.f;

    // Optional HUD-owned status line (e.g. for enrollment messages)
    std::string statusOverride_;

    // Right arrow toggles the HUD panel (menu) on/off
    bool     panelVisible_         = true;

    // Multi-layer state
    UIPanel  uiPanel_             = UIPanel::Main;
    bool     detectionActive_      = false;
    bool     faceDetectionActive_  = false;
    bool     voiceAssistActive_    = false;
    int      m_stripSelectedIndex  = 0;
    int      selectedMainIndex_    = 0;  // 0=Object Detection .. 5=Settings (keyboard selection)
    int      selectedTranslationIndex_ = 0;  // 0..4 in Translation Hub (OCR, Speech->Text, Text->Speech, Sign, Back)
    int      selectedVoiceAssistIndex_ = 0;   // 0=Voice Feedback, 1=Voice Assist toggle, 2=Back
    int      selectedAddDataIndex_     = 0;   // 0=Add Face, 1=Add Object, 2=Back
    int      selectedSettingsIndex_    = 0;   // 0=Back only

    // Pending action to be polled by main.cpp
    HUDAction pendingAction_;

    // Current task when draw() ran; taskMgr set each frame for handleInput (Escape, overlay)
    TranslationTask lastKnownTask_ = TranslationTask::NONE;
    TranslationTaskManager* taskMgr_ = nullptr;

    // One-shot flags for Text->Speech overlay:
    //  - ttsInputFocusRequested_: focus the input box once on mode entry
    //  - ttsOverlayJustOpened_:   apply window pos/size once on mode entry
    bool ttsInputFocusRequested_ = false;
    bool ttsOverlayJustOpened_   = false;


    // Renders face detection boxes, landmarks, pose labels onto frame
    void drawFaces(cv::Mat& frame, const std::vector<FaceDetector::FaceResult>& faces);

    // Face enrollment (Add Face) state machine
    EnrollState     enrollState_        = EnrollState::IDLE;
    char            enrollNameBuf_[64]  = {};
    std::string     enrolledName_;
    std::vector<cv::Mat> capturedFaces_;
    bool            enrollNameFocusSet_ = false;

    // Face recognition backend (optional)
    std::unique_ptr<hud::FaceEncoder>    faceEncoder_;
    std::unique_ptr<hud::IdentityStore>  identityStore_;

    // Helpers for enrollment
    void updateEnrollment(const cv::Mat& frame, const std::vector<cv::Rect>& faceRects);
    std::vector<float> buildEmbeddingFromCapturedFaces();

    // ── Task [10]: New overlay methods (read from gResultsStore) ──────────
    void drawOCRRegionOverlays(float vX, float vY, float vW, float vH, int captureW, int captureH,
                               TranslationTaskManager* tm);
    void drawOCRCaptureButton(float vW, float vH, float yOff, TranslationTaskManager* tm);
    void drawOCROutputIcons (float vW, float vH, float yOff);
    void drawSignOutputIcons(float vW, float vH, float yOff);
    void drawOCROverlay     (float vW, float vH, float yOff);
    void drawSTTSubtitle    (float vW, float vH, float yOff, bool showTranslation, float bottomReserved = 0.f);
    void drawTTSOverlay     (float vW, float vH, float yOff, float bottomReserved = 0.f);
    void drawTTSIndicator();
    void drawResolutionOverlay(float vW, float vH);
    void drawSTTTTSInputOverlay(float vW, float vH, float yOff, TranslationTaskManager* tm);

    // ── HUD mode toggle + minimized strip ──────────────────────────────────
    bool m_minimizedMode = true; // default: new circular HUD
    bool m_stripVisible  = true;
    void drawMinimizedTaskStrip();
    void drawMinimizedSubPanel();
    void drawMinimizedAddDataPanel();
    void drawMinimizedSettingsPanel();
    void drawNodeIcon(ImDrawList* dl, const ImVec2& center, UIPanel panel, ImU32 col, float size);
    void drawSubOptionIcon(ImDrawList* dl, ImVec2 c, UIPanel panel, int optionIndex, ImU32 col, float size);

    static constexpr int kNodeCount = 6;
    float  m_nodeRadius[kNodeCount]     = {7.f, 7.f, 7.f, 7.f, 7.f, 7.f};
    bool   m_wasHovered[kNodeCount]     = {};
    double m_hoverStartTime[kNodeCount] = {};

    // ── Minimized sub-panel (2nd layer) state ───────────────────────────────
    int   m_subpanelSelectedIndex = 0;
    float m_subpanelRadius[16]    = {};
};

} // namespace SmartGlasses



