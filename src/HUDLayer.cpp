/*
 * HUDLayer.cpp Ã¢â‚¬â€œ Smart Glasses HUD
 * Fixes:
 *   Ã¢â‚¬Â¢ BGR -> RGB conversion before GL texture upload (black screen fix)
 *   Ã¢â‚¬Â¢ 15% right panel, full height, multi-layer task navigation
 *   Ã¢â‚¬Â¢ Header bar shows FPS, status, hint, confidence
 *   Ã¢â‚¬Â¢ Detection boxes + face rects drawn on video with ImDrawList
 */

// Must come before ANY other include on Windows to prevent min/max macro
// conflicts with OpenCV (C2589 error on imgproc.hpp)
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

#include "HUDLayer.hpp"
#include "ModelRegistry.hpp"
#include "TranslationTaskManager.hpp"
#include "Config.hpp"
#include "ResultsStore.h"
#include "STTEngine.h"
#include "TTSEngine.h"
#include "OCREngine.h"
#include "TranslationQueue.h"
#include "face/face_encoder.h"
#include "face/identity_store.h"
#include <fstream>
#include <cmath>
#include <spdlog/spdlog.h>

// ── Async STT/TTS state (prevents HUD freeze) ───────────────────────────
#include <thread>
#include <atomic>
#include <mutex>
static std::atomic<bool>  g_sttRunning{false};
static std::atomic<bool>  g_ttsRunning{false};
static std::string         g_sttResult;
static std::mutex          g_sttMutex;

static void asyncSTT(int seconds) {
    if (g_sttRunning.exchange(true)) return;
    std::thread([seconds](){
        std::string result = SmartGlasses::asyncSTT(seconds);
        { std::lock_guard<std::mutex> lk(g_sttMutex); g_sttResult = result; }
        gResultsStore.setSTTOriginal(result, "en");
        gResultsStore.setSTTStatus("ready");
        g_sttRunning = false;
    }).detach();
}
// ─────────────────────────────────────────────────────────────────────────

// ── Bounding box temporal smoother ──────────────────────────────────────
#include <map>
struct SmoothedBox { float x,y,w,h; int ttl; };
static std::map<int, SmoothedBox> g_smoothedBoxes;
static cv::Rect smoothBox(int id, const cv::Rect& raw, float alpha = 0.35f) {
    auto it = g_smoothedBoxes.find(id);
    if (it == g_smoothedBoxes.end()) {
        g_smoothedBoxes[id] = {(float)raw.x,(float)raw.y,(float)raw.width,(float)raw.height, 3};
        return raw;
    }
    auto& s = it->second;
    s.x = s.x + alpha*(raw.x - s.x);
    s.y = s.y + alpha*(raw.y - s.y);
    s.w = s.w + alpha*(raw.width  - s.w);
    s.h = s.h + alpha*(raw.height - s.h);
    s.ttl = 4;
    return cv::Rect((int)s.x,(int)s.y,(int)s.w,(int)s.h);
}
static void decaySmoothedBoxes() {
    for (auto it = g_smoothedBoxes.begin(); it != g_smoothedBoxes.end(); ) {
        if (--it->second.ttl <= 0) it = g_smoothedBoxes.erase(it);
        else ++it;
    }
}
// ────────────────────────────────────────────────────────────────────────

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

namespace SmartGlasses {

extern GLFWwindow* gHudWindow;

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Colour palette Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
static const ImVec4 kCyan      = {0.00f, 0.85f, 0.90f, 1.f};
static const ImVec4 kCyanDim   = {0.00f, 0.55f, 0.60f, 1.f};
static const ImVec4 kGreen     = {0.20f, 0.90f, 0.40f, 1.f};
static const ImVec4 kRed       = {0.90f, 0.20f, 0.20f, 1.f};
static const ImVec4 kOrange    = {0.95f, 0.55f, 0.05f, 1.f};
static const ImVec4 kPanelBg   = {0.04f, 0.06f, 0.09f, 0.96f};
static const ImVec4 kHeaderBg  = {0.02f, 0.04f, 0.07f, 0.92f};
static const ImVec4 kBtnNorm   = {0.08f, 0.14f, 0.20f, 1.f};
static const ImVec4 kBtnHover  = {0.10f, 0.25f, 0.35f, 1.f};
static const ImVec4 kBtnActive = {0.00f, 0.55f, 0.65f, 1.f};
static const ImVec4 kBtnOn     = {0.00f, 0.45f, 0.55f, 1.f};  // "live" state

static constexpr float PANEL_FRAC   = 0.15f;  // right panel = 15% of window width
static constexpr float HEADER_H     = 32.f;

static const char* layerName(int idx) {
    switch (idx) {
        case 0: return "Detection View";
        case 1: return "Translation View";
        case 2: return "Face Recognition View";
        case 3: return "Full Overlay View";
        case 4: return "Settings View";
        default: return "Unknown";
    }
}

static std::string getRegionLabel(const cv::Rect& r) {
    if (r.width < 100)       return "Word";
    if (r.width <= 300)      return "Text";
    return "Sign / Board";
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Constructor / Destructor Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
HUDLayer::HUDLayer()  {
    try {
        faceEncoder_ = std::make_unique<hud::FaceEncoder>(Config::modelsDir);
    } catch (const std::exception& e) {
        spdlog::warn("[HUD] FaceEncoder init failed: {}", e.what());
    }
    try {
        identityStore_ = std::make_unique<hud::IdentityStore>("data/identities/identities.json");
    } catch (const std::exception& e) {
        spdlog::warn("[HUD] IdentityStore init failed: {}", e.what());
    }
}
HUDLayer::~HUDLayer() {
    if (cameraTexID_) glDeleteTextures(1, &cameraTexID_);
}

// Face enrollment helper: capture frames and run encoding when 50 samples collected.
void HUDLayer::updateEnrollment(const cv::Mat& frame, const std::vector<cv::Rect>& faceRects) {
    if (enrollState_ != EnrollState::CAPTURING)
        return;
    if (frame.empty())
        return;

    if (!faceRects.empty()) {
        cv::Rect box = faceRects.front() & cv::Rect(0, 0, frame.cols, frame.rows);
        if (box.width > 0 && box.height > 0) {
            capturedFaces_.push_back(frame(box).clone());
            std::cout << "[Enroll] Captured frame " << capturedFaces_.size() << "/50\n";
        }
    }

    if (capturedFaces_.size() >= 50) {
        enrollState_ = EnrollState::ENCODING;
        statusOverride_ = "Encoding face for: " + enrolledName_ + " ...";
        std::cout << "[Enroll] Reached 50 frames, starting encoding for " << enrolledName_ << "\n";

        if (faceEncoder_ && identityStore_ && !enrolledName_.empty()) {
            auto emb = buildEmbeddingFromCapturedFaces();
            if (!emb.empty()) {
                identityStore_->AddIdentity(enrolledName_, emb);
                statusOverride_ = "Identity saved: " + enrolledName_;
                std::cout << "[Enroll] Identity saved for " << enrolledName_ << "\n";
            } else {
                statusOverride_ = "Encoding failed for: " + enrolledName_;
                std::cout << "[Enroll] Encoding failed for " << enrolledName_ << "\n";
            }
        } else {
            statusOverride_ = "Enrollment backend unavailable";
            std::cout << "[Enroll] Backend unavailable for enrollment\n";
        }

        capturedFaces_.clear(); // delete all captured frames from memory
        enrollState_ = EnrollState::DONE;
    }
}

std::vector<float> HUDLayer::buildEmbeddingFromCapturedFaces() {
    std::vector<float> avg(512, 0.0f);
    if (!faceEncoder_)
        return {};
    int valid = 0;
    for (auto& img : capturedFaces_) {
        auto emb = faceEncoder_->Encode(img);
        if (emb.size() != 512)
            continue;
        if (valid == 0)
            avg.assign(emb.begin(), emb.end());
        else {
            for (int i = 0; i < 512; ++i)
                avg[i] += emb[i];
        }
        ++valid;
    }
    if (valid == 0)
        return {};
    for (auto& v : avg)
        v /= static_cast<float>(valid);
    hud::FaceEncoder::Normalize(avg);
    return avg;
}

void HUDLayer::handleInput() {
    ImGuiIO& io = ImGui::GetIO();

    const bool keyUp    = ImGui::IsKeyPressed(ImGuiKey_UpArrow,    false);
    const bool keyDown  = ImGui::IsKeyPressed(ImGuiKey_DownArrow,  false);
    const bool keyRight = ImGui::IsKeyPressed(ImGuiKey_RightArrow, false);
    const bool keyLeft  = ImGui::IsKeyPressed(ImGuiKey_LeftArrow,  false);
    const bool keyEnter = ImGui::IsKeyPressed(ImGuiKey_Enter,      false);
    const bool doActivate = keyRight || keyEnter;

    // Ctrl+Up/Down: move HUD position (always fires)
    if (keyUp && io.KeyCtrl) {
        Config::hudPositionY =
            Config::clampHudY(Config::hudPositionY - Config::HUD_MOVE_STEP);
        return;
    }
    if (keyDown && io.KeyCtrl) {
        Config::hudPositionY =
            Config::clampHudY(Config::hudPositionY + Config::HUD_MOVE_STEP);
        return;
    }

    if (m_minimizedMode) {
        // Left Arrow: toggle strip visibility or go back one level
        if (keyLeft) {
            if (uiPanel_ != UIPanel::Main) {
                uiPanel_ = UIPanel::Main;
                m_subpanelSelectedIndex = 0;
            } else {
                m_stripVisible = !m_stripVisible;
            }
            return;
        }

        // Layer 1
        if (uiPanel_ == UIPanel::Main) {
            if (!m_stripVisible) return;
            if (keyDown) {
                m_stripSelectedIndex =
                    (m_stripSelectedIndex + 1) % kNodeCount;
                return;
            }
            if (keyUp) {
                m_stripSelectedIndex =
                    (m_stripSelectedIndex + kNodeCount - 1) % kNodeCount;
                return;
            }
            if (keyRight) {
                static const UIPanel kMap[kNodeCount] = {
                    UIPanel::ObjectDetection,
                    UIPanel::FaceDetection,
                    UIPanel::Translation,
                    UIPanel::Speech,
                    UIPanel::AddData,
                    UIPanel::Settings,
                };

                UIPanel selected = kMap[m_stripSelectedIndex];

                // These tasks activate directly — NO second layer
                if (selected == UIPanel::ObjectDetection) {
                    detectionActive_ = !detectionActive_;
                    if (detectionActive_ && taskMgr_)
                        taskMgr_->setTask(TranslationTask::SCENE_TO_SPEECH);
                    return;
                }

                if (selected == UIPanel::FaceDetection) {
                    faceDetectionActive_ = !faceDetectionActive_;
                    if (faceDetectionActive_ && taskMgr_)
                        taskMgr_->setTask(TranslationTask::FACE_TO_SPEECH);
                    return;
                }

                // These have second layers
                if (selected == UIPanel::Translation ||
                    selected == UIPanel::AddData ||
                    selected == UIPanel::Settings ||
                    selected == UIPanel::Speech) {
                    uiPanel_ = selected;
                    m_subpanelSelectedIndex = 0;
                    return;
                }
            }
            return;
        }

        // Layer 2
        {
            int optionCount = 0;
            switch (uiPanel_) {
                case UIPanel::Translation:     optionCount = 4; break;
                case UIPanel::Speech:          optionCount = 1; break;
                default:                       optionCount = 1; break;
            }

            if (keyDown) {
                m_subpanelSelectedIndex =
                    (m_subpanelSelectedIndex + 1) % optionCount;
                return;
            }
            if (keyUp) {
                m_subpanelSelectedIndex =
                    (m_subpanelSelectedIndex + optionCount - 1) % optionCount;
                return;
            }

            if (doActivate) {
                const int id = m_subpanelSelectedIndex;
                switch (uiPanel_) {
                    case UIPanel::Translation:
                        if (id == 0) {
                            // OCR selection (no nested layer here)
                            pendingAction_.type = HUDActionType::CaptureOCR;
                            pendingAction_.taskIdx = -1;
                        } else if (id == 1 && taskMgr_) {
                            taskMgr_->setTask(
                                TranslationTask::SPEECH_TO_TEXT);
                        } else if (id == 2 && taskMgr_) {
                            taskMgr_->setTask(
                                TranslationTask::TEXT_TO_SPEECH);
                        } else if (id == 3) {
                            // Sign selection (no nested layer here)
                            if (taskMgr_)
                                taskMgr_->setTask(TranslationTask::SIGN_TO_TEXT);
                        }
                        break;

                    case UIPanel::Speech:
                        if (id == 0) {
                            if (taskMgr_)
                                taskMgr_->setTask(TranslationTask::SPEECH_TO_TEXT);
                        } else if (id == 1) {
                            if (taskMgr_)
                                taskMgr_->setTask(TranslationTask::TEXT_TO_SPEECH);
                        }
                        break;

                    default: break;
                }
                return;
            }
            return;
        }
    }
    // Non-minimized mode falls through to existing handleInput() code below

    // When TTS input has focus, Enter submits text; we still allow keyboard navigation.
    const bool translationOverlayActive =
        (lastKnownTask_ != TranslationTask::NONE &&
         lastKnownTask_ != TranslationTask::TEXT_TO_SPEECH);

    // Raw Enter/KeypadEnter using GLFW to bypass ImGui input routing entirely.
    bool enterDown = false;
    if (gHudWindow) {
        enterDown =
            glfwGetKey(gHudWindow, GLFW_KEY_ENTER)     == GLFW_PRESS ||
            glfwGetKey(gHudWindow, GLFW_KEY_KP_ENTER)  == GLFW_PRESS;
    }
    static bool s_enterWasDown = false;
    // In S2T mode Space is reserved for start/stop listening; don't use it as Enter.
    bool enterOnly = false;
    if (gHudWindow)
        enterOnly = (glfwGetKey(gHudWindow, GLFW_KEY_ENTER) == GLFW_PRESS || glfwGetKey(gHudWindow, GLFW_KEY_KP_ENTER) == GLFW_PRESS);
    const bool enterDownForPanel = (lastKnownTask_ == TranslationTask::SPEECH_TO_TEXT) ? enterOnly : enterDown;
    const bool doEnter = enterDownForPanel && !s_enterWasDown;
    s_enterWasDown = enterDownForPanel;

    // When Text->Speech input box has focus, let it consume Enter so the user can submit text.
    const bool textInputFocused =
        ImGui::IsAnyItemActive() && (lastKnownTask_ == TranslationTask::TEXT_TO_SPEECH);

    auto goBack = [&]() {
        if (uiPanel_ != UIPanel::Main)
            uiPanel_ = UIPanel::Main;
    };

    auto activateMainSelection = [&]() {
        if (uiPanel_ != UIPanel::Main) return;
        switch (selectedMainIndex_) {
            case 0:
                detectionActive_ = !detectionActive_;
                pendingAction_.type = detectionActive_ ? HUDActionType::ActivateDetection : HUDActionType::DeactivateDetection;
                pendingAction_.taskIdx = -1;
                break;
            case 1:
                faceDetectionActive_ = !faceDetectionActive_;
                pendingAction_.type = faceDetectionActive_ ? HUDActionType::StartFaceDetection : HUDActionType::StopFaceDetection;
                pendingAction_.taskIdx = -1;
                break;
            case 2: uiPanel_ = UIPanel::Translation; break;
            case 3: uiPanel_ = UIPanel::VoiceAssist; break;
            case 4: uiPanel_ = UIPanel::AddData; break;
            case 5: uiPanel_ = UIPanel::Settings; break;
            default: break;
        }
    };

    constexpr int TRANSLATION_OPTIONS = 5;
    auto activateTranslationSelection = [&]() {
        if (uiPanel_ != UIPanel::Translation) return;
        if (selectedTranslationIndex_ == 4) { uiPanel_ = UIPanel::Main; return; }
        static const TranslationTask kTasks[] = {
            TranslationTask::OCR_TO_TEXT,   // merged OCR (output mode via icons)
            TranslationTask::SPEECH_TO_TEXT,
            TranslationTask::TEXT_TO_SPEECH,
            TranslationTask::SIGN_TO_TEXT,   // merged Sign (output mode via icons)
        };
        pendingAction_.type = HUDActionType::SetTask;
        pendingAction_.taskIdx = static_cast<int>(kTasks[selectedTranslationIndex_]);
    };

    constexpr int VOICEASSIST_OPTIONS = 3;
    auto activateVoiceAssistSelection = [&]() {
        if (uiPanel_ != UIPanel::VoiceAssist) return;
        if (selectedVoiceAssistIndex_ == 2) { uiPanel_ = UIPanel::Main; return; }
        if (selectedVoiceAssistIndex_ == 0) {
            voiceFeedback_ = !voiceFeedback_;
            if (Config::TTS_ENABLED != voiceFeedback_) Config::TTS_ENABLED = voiceFeedback_;
        } else if (selectedVoiceAssistIndex_ == 1) {
            voiceAssistActive_ = !voiceAssistActive_;
            pendingAction_ = { voiceAssistActive_ ? HUDActionType::StartVoiceAssist : HUDActionType::StopVoiceAssist, -1 };
        }
    };

    constexpr int ADDDATA_OPTIONS = 3;
    auto activateAddDataSelection = [&]() {
        if (uiPanel_ != UIPanel::AddData) return;
        if (selectedAddDataIndex_ == 2) { uiPanel_ = UIPanel::Main; return; }
        if (selectedAddDataIndex_ == 0) {
            if (enrollState_ == EnrollState::IDLE || enrollState_ == EnrollState::DONE) {
                enrollState_ = EnrollState::WAITING_NAME;
                std::memset(enrollNameBuf_, 0, sizeof(enrollNameBuf_));
                enrollNameFocusSet_ = false;
                capturedFaces_.clear();
                statusOverride_.clear();
                std::cout << "[Enroll] Entered WAITING_NAME state\n";
            }
        } else if (selectedAddDataIndex_ == 1) {
            std::cout << "[HUD] AddData: dispatch AddObject\n";
            pendingAction_ = { HUDActionType::AddObject, -1 };
        }
    };

    auto activateSettingsSelection = [&]() {
        if (uiPanel_ != UIPanel::Settings) return;
        uiPanel_ = UIPanel::Main;
    };

    // Arrow keys + Enter only (no Escape, F2, F3)
    // Process keys even when a button has focus so Enter always activates

    // Ctrl+Up/Down: HUD position (always, even when typing in TTS input)
    if (keyUp && io.KeyCtrl) {
        Config::hudPositionY = Config::clampHudY(Config::hudPositionY - Config::HUD_MOVE_STEP);
        return;
    }
    if (keyDown && io.KeyCtrl) {
        Config::hudPositionY = Config::clampHudY(Config::hudPositionY + Config::HUD_MOVE_STEP);
        return;
    }

    // Layer 2: translation overlay active â€” Up/Down + Enter only; Left/Right pass through
    // When a text input (e.g. TTS) has focus, skip Enter so it can submit; we still handle arrows
    if (translationOverlayActive) {
        if (keyUp) {
            if (uiPanel_ == UIPanel::Main)
                selectedMainIndex_ = (selectedMainIndex_ - 1 + 6) % 6;
            else if (uiPanel_ == UIPanel::Translation)
                selectedTranslationIndex_ = (selectedTranslationIndex_ - 1 + TRANSLATION_OPTIONS) % TRANSLATION_OPTIONS;
            else if (uiPanel_ == UIPanel::VoiceAssist)
                selectedVoiceAssistIndex_ = (selectedVoiceAssistIndex_ - 1 + VOICEASSIST_OPTIONS) % VOICEASSIST_OPTIONS;
            else if (uiPanel_ == UIPanel::AddData)
                selectedAddDataIndex_ = (selectedAddDataIndex_ - 1 + ADDDATA_OPTIONS) % ADDDATA_OPTIONS;
        }
        if (keyDown) {
            if (uiPanel_ == UIPanel::Main)
                selectedMainIndex_ = (selectedMainIndex_ + 1) % 6;
            else if (uiPanel_ == UIPanel::Translation)
                selectedTranslationIndex_ = (selectedTranslationIndex_ + 1) % TRANSLATION_OPTIONS;
            else if (uiPanel_ == UIPanel::VoiceAssist)
                selectedVoiceAssistIndex_ = (selectedVoiceAssistIndex_ + 1) % VOICEASSIST_OPTIONS;
            else if (uiPanel_ == UIPanel::AddData)
                selectedAddDataIndex_ = (selectedAddDataIndex_ + 1) % ADDDATA_OPTIONS;
        }
        if (doEnter && !textInputFocused) {
            if (uiPanel_ == UIPanel::Translation)
                std::cout << "[HUD] ENTER FIRED: Translation idx=" << selectedTranslationIndex_ << "\n";
            if (uiPanel_ == UIPanel::AddData)
                std::cout << "[HUD] ENTER FIRED: AddData idx=" << selectedAddDataIndex_ << "\n";
            if (uiPanel_ == UIPanel::Main) activateMainSelection();
            else if (uiPanel_ == UIPanel::Translation) activateTranslationSelection();
            else if (uiPanel_ == UIPanel::VoiceAssist) activateVoiceAssistSelection();
            else if (uiPanel_ == UIPanel::AddData) activateAddDataSelection();
            else if (uiPanel_ == UIPanel::Settings) activateSettingsSelection();
        }
        if (keyLeft && !io.KeyCtrl)
            panelVisible_ = !panelVisible_;
        if (keyRight && !io.KeyCtrl) {
            if (uiPanel_ == UIPanel::Main) activateMainSelection();
            else if (uiPanel_ == UIPanel::Translation) activateTranslationSelection();
            else if (uiPanel_ == UIPanel::VoiceAssist) activateVoiceAssistSelection();
            else if (uiPanel_ == UIPanel::AddData) activateAddDataSelection();
            else if (uiPanel_ == UIPanel::Settings) activateSettingsSelection();
        }
        return;  // do not handle more Left/Right when overlay active
    }

    // Layer 3: Left = HUD panel on/off, Right = current task on/off (activate selection)
    if (keyLeft) {
        if (io.KeyCtrl)
            Config::activeLayer = (Config::activeLayer - 1 + Config::TOTAL_HUD_LAYERS) % Config::TOTAL_HUD_LAYERS;
        else
            panelVisible_ = !panelVisible_;  // Left: HUD on/off
    }
    if (keyRight) {
        if (io.KeyCtrl)
            Config::activeLayer = (Config::activeLayer + 1) % Config::TOTAL_HUD_LAYERS;
        else if (uiPanel_ == UIPanel::Main)
            activateMainSelection();
        else if (uiPanel_ == UIPanel::Translation)
            activateTranslationSelection();
        else if (uiPanel_ == UIPanel::VoiceAssist)
            activateVoiceAssistSelection();
        else if (uiPanel_ == UIPanel::AddData)
            activateAddDataSelection();
        else if (uiPanel_ == UIPanel::Settings)
            activateSettingsSelection();
    }
    if (keyUp) {
        if (uiPanel_ == UIPanel::Main) selectedMainIndex_ = (selectedMainIndex_ - 1 + 6) % 6;
        else if (uiPanel_ == UIPanel::Translation) selectedTranslationIndex_ = (selectedTranslationIndex_ - 1 + TRANSLATION_OPTIONS) % TRANSLATION_OPTIONS;
        else if (uiPanel_ == UIPanel::VoiceAssist) selectedVoiceAssistIndex_ = (selectedVoiceAssistIndex_ - 1 + VOICEASSIST_OPTIONS) % VOICEASSIST_OPTIONS;
        else if (uiPanel_ == UIPanel::AddData) selectedAddDataIndex_ = (selectedAddDataIndex_ - 1 + ADDDATA_OPTIONS) % ADDDATA_OPTIONS;
    }
    if (keyDown) {
        if (uiPanel_ == UIPanel::Main) selectedMainIndex_ = (selectedMainIndex_ + 1) % 6;
        else if (uiPanel_ == UIPanel::Translation) selectedTranslationIndex_ = (selectedTranslationIndex_ + 1) % TRANSLATION_OPTIONS;
        else if (uiPanel_ == UIPanel::VoiceAssist) selectedVoiceAssistIndex_ = (selectedVoiceAssistIndex_ + 1) % VOICEASSIST_OPTIONS;
        else if (uiPanel_ == UIPanel::AddData) selectedAddDataIndex_ = (selectedAddDataIndex_ + 1) % ADDDATA_OPTIONS;
    }
    if (doEnter && !textInputFocused) {
        if (uiPanel_ == UIPanel::Translation)
            std::cout << "[HUD] ENTER FIRED: Translation idx=" << selectedTranslationIndex_ << "\n";
        if (uiPanel_ == UIPanel::AddData)
            std::cout << "[HUD] ENTER FIRED: AddData idx=" << selectedAddDataIndex_ << "\n";
        if (uiPanel_ == UIPanel::Main) activateMainSelection();
        else if (uiPanel_ == UIPanel::Translation) activateTranslationSelection();
        else if (uiPanel_ == UIPanel::VoiceAssist) activateVoiceAssistSelection();
        else if (uiPanel_ == UIPanel::AddData) activateAddDataSelection();
        else if (uiPanel_ == UIPanel::Settings) activateSettingsSelection();
    }
}

void HUDLayer::drawLayerIndicator(float displayW) {
    // Fixed widget: not affected by hudPositionY
    constexpr float padX = 12.f;
    constexpr float padY = 10.f;

    const char* name = layerName(Config::activeLayer);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "â— Layer %d/%d â€” %s",
                  Config::activeLayer + 1, Config::TOTAL_HUD_LAYERS, name);

    ImVec2 textSz = ImGui::CalcTextSize(buf);
    ImVec2 hintSz = ImGui::CalcTextSize("â† â†’");
    float winW = textSz.x + hintSz.x + 34.f;
    float winH = 26.f;

    ImGui::SetNextWindowPos(ImVec2(displayW - winW - padX, padY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(winW, winH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.35f);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.35f));
    ImGui::PushStyleColor(ImGuiCol_Border,   kCyanDim);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.f, 6.f));

    ImGui::Begin("##layer_indicator", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoInputs |
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::PushStyleColor(ImGuiCol_Text, kCyan);
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();

    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, kCyanDim);
    ImGui::TextUnformatted("  â† â†’");
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ uploadFrame: BGR -> RGB, then upload to GL texture Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
GLuint HUDLayer::uploadFrame(const cv::Mat& frame) {
    if (frame.empty()) return 0;

    // Convert BGR (OpenCV default) -> RGB (OpenGL expects)
    cv::Mat rgb;
    if (frame.channels() == 3)
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    else if (frame.channels() == 4)
        cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGBA);
    else
        rgb = frame; // grayscale or other; upload as-is

    const GLenum fmt = (rgb.channels() == 4) ? GL_RGBA : GL_RGB;

    if (cameraTexID_ == 0) {
        glGenTextures(1, &cameraTexID_);
        glBindTexture(GL_TEXTURE_2D, cameraTexID_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
        glBindTexture(GL_TEXTURE_2D, cameraTexID_);
    }

    if (rgb.cols != lastWidth_ || rgb.rows != lastHeight_) {
        // (Re)allocate texture storage
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     rgb.cols, rgb.rows, 0,
                     fmt, GL_UNSIGNED_BYTE, rgb.data);
        lastWidth_  = rgb.cols;
        lastHeight_ = rgb.rows;
    } else {
        // Fast sub-image update (no reallocation)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        rgb.cols, rgb.rows,
                        fmt, GL_UNSIGNED_BYTE, rgb.data);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    return cameraTexID_;
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ pollAction Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
HUDAction HUDLayer::pollAction() {
    HUDAction a = pendingAction_;
    pendingAction_.type = HUDActionType::None;
    pendingAction_.taskIdx = -1;
    pendingAction_.text.clear();
    pendingAction_.ocrClickX = -1;
    pendingAction_.ocrClickY = -1;
    return a;
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Helper: full-width styled button Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
bool HUDLayer::panelButton(const char* label, bool active) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 size  = {avail.x, 34.f};

    // Hover/active use the same color so there is never a second
    // â€œmouse hoverâ€ highlight on top of the keyboard selection.
    ImVec4 base = active ? kBtnOn : kBtnNorm;
    ImGui::PushStyleColor(ImGuiCol_Button,        base);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, base);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  base);
    bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);
    ImGui::Spacing();
    return clicked;
}

// Active = task on (filled), Selected = keyboard focus (border) â€” two distinct highlights
bool HUDLayer::panelButtonEx(const char* label, bool isActive, bool isSelected) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 size  = {avail.x, 34.f};

    ImVec4 col = isActive ? kBtnOn : (isSelected ? kBtnHover : kBtnNorm);
    // Same color for normal / hovered / active so only the cyan border
    // (isSelected) defines the highlight â€“ no second hover highlight.
    ImGui::PushStyleColor(ImGuiCol_Button,        col);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, col);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  col);

    // Keep ImGui's keyboard focus (blue outline) in sync with our
    // custom selection index when the user holds Up/Down.
    if (isSelected)
        ImGui::SetKeyboardFocusHere();

    if (isSelected)
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.f);
    bool clicked = ImGui::Button(label, size);
    if (isSelected)
        ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);
    if (isSelected) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        if (dl) {
            ImVec2 a = ImGui::GetItemRectMin();
            ImVec2 b = ImGui::GetItemRectMax();
            dl->AddRect(a, b, ImGui::ColorConvertFloat4ToU32(kCyan), 0.f, 0, 2.f);
        }
    }
    ImGui::Spacing();
    return clicked;
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Sub-panel: Main Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawPanel_Main(TranslationTaskManager* tm) {
    ImGui::PushStyleColor(ImGuiCol_Text, kCyan);
    ImGui::TextUnformatted("  SELECT TASK");
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::Spacing();

    // Only ONE highlight: keyboard-selected item (arrow keys). Task-on shown with â— suffix.
    // Object Detection: blue highlight (active) instead of text marker.
    if (panelButtonEx("  Object Detection", detectionActive_, selectedMainIndex_ == 0)) {
        selectedMainIndex_ = 0;
        detectionActive_ = !detectionActive_;
        if (tm) {
            if (detectionActive_) {
                tm->setTask(TranslationTask::SCENE_TO_SPEECH);
                pendingAction_ = {HUDActionType::ActivateDetection, -1};
            } else {
                pendingAction_ = {HUDActionType::DeactivateDetection, -1};
            }
        }
    }

    // Face Detection: blue highlight (active) instead of text marker.
    if (panelButtonEx("  Face Detection", faceDetectionActive_, selectedMainIndex_ == 1)) {
        selectedMainIndex_ = 1;
        faceDetectionActive_ = !faceDetectionActive_;
        if (tm) {
            if (faceDetectionActive_) {
                tm->setTask(TranslationTask::FACE_TO_SPEECH);
                pendingAction_ = {HUDActionType::StartFaceDetection, -1};
            } else {
                pendingAction_ = {HUDActionType::StopFaceDetection, -1};
            }
        }
    }
    if (panelButtonEx("  Translation Hub", false, selectedMainIndex_ == 2)) {
        selectedMainIndex_ = 2;
        uiPanel_ = UIPanel::Translation;
    }
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "  Voice Assist%s", voiceAssistActive_ ? "  â—" : "");
        if (panelButtonEx(buf, false, selectedMainIndex_ == 3)) {
            selectedMainIndex_ = 3;
            uiPanel_ = UIPanel::VoiceAssist;
        }
    }
    if (panelButtonEx("  Add Data", false, selectedMainIndex_ == 4)) {
        selectedMainIndex_ = 4;
        uiPanel_ = UIPanel::AddData;
    }
    if (panelButtonEx("  Settings", false, selectedMainIndex_ == 5)) {
        selectedMainIndex_ = 5;
        uiPanel_ = UIPanel::Settings;
    }
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Sub-panel: Object Detection Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawPanel_ObjectDetection(TranslationTaskManager* tm) {
    float panelW = 220.f;
    float panelH = 320.f;
    float screenW = ImGui::GetIO().DisplaySize.x;
    float screenH = ImGui::GetIO().DisplaySize.y;

    ImGui::SetNextWindowPos(
        ImVec2(screenW - panelW - 20.f, screenH * 0.12f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH),
                               ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
    ImGui::Begin("##panel_object_detection", nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddText(ImVec2(wp.x, wp.y + 8.f),
                IM_COL32(0, 230, 200, 220), "OBJECT DETECTION");
    dl->AddLine(ImVec2(wp.x, wp.y + 26.f),
                 ImVec2(wp.x + panelW, wp.y + 26.f),
                 IM_COL32(0, 200, 180, 80), 1.0f);

    ImGui::SetCursorPosY(38.f);

    auto techButton = [&](const char* label, const char* id, bool isActive) -> bool {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.35f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                               ImVec4(0.07f, 0.43f, 0.9f, 0.55f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                               ImVec4(0.1f, 0.55f, 1.f, 0.85f));
        ImGui::PushStyleColor(ImGuiCol_Text,
                               ImVec4(0.85f, 0.95f, 1.f, 1.f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.f);
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.f, 0.8f, 0.7f, 0.3f));

        if (isActive) {
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                   ImVec4(0.05f, 0.35f, 0.85f, 0.75f));
            ImGui::PushStyleColor(ImGuiCol_Button,
                                   ImVec4(0.08f, 0.45f, 1.0f, 0.90f));
            // Update ButtonActive to match filled state too
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                   ImVec4(0.08f, 0.45f, 1.0f, 0.90f));
        }

        ImGui::SetNextItemWidth(-1.f);
        std::string full = std::string(label) + "##" + id;
        bool clicked = ImGui::Button(full.c_str(), ImVec2(-1.f, 32.f));

        if (isActive) ImGui::PopStyleColor(3);
        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);
        return clicked;
    };

    // Voice feedback toggle
    ImGui::Spacing();
    if (voiceFeedback_) {
        if (techButton("VOICE: ON", "voice_on", true)) {
            voiceFeedback_ = false;
            if (Config::TTS_ENABLED != voiceFeedback_) Config::TTS_ENABLED = voiceFeedback_;
        }
    } else {
        if (techButton("VOICE: OFF", "voice_off", true)) {
            voiceFeedback_ = true;
            if (Config::TTS_ENABLED != voiceFeedback_) Config::TTS_ENABLED = voiceFeedback_;
        }
    }
    ImGui::Spacing();
    ImGui::Spacing();

    // Enable / Disable
    if (techButton("ENABLE", "enable", detectionActive_)) {
        detectionActive_ = true;
        if (tm) tm->setTask(TranslationTask::SCENE_TO_SPEECH);
        pendingAction_ = {HUDActionType::ActivateDetection, -1};
        ImGui::Spacing();
    }
    ImGui::Spacing();
    ImGui::Spacing();

    if (techButton("DISABLE", "disable", !detectionActive_)) {
        detectionActive_ = false;
        pendingAction_ = {HUDActionType::DeactivateDetection, -1};
        ImGui::Spacing();
    }
    ImGui::Spacing();
    ImGui::Spacing();

    // Status text
    ImGui::PushStyleColor(ImGuiCol_Text,
                           ImVec4(0.f, 0.9f, 0.75f, 0.9f));
    ImGui::Text(detectionActive_ ? "RUNNING" : "IDLE");
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Sub-panel: Face Detection Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawPanel_FaceDetection(TranslationTaskManager* tm) {
    (void)tm;

    float panelW = 220.f;
    float panelH = 320.f;
    float screenW = ImGui::GetIO().DisplaySize.x;
    float screenH = ImGui::GetIO().DisplaySize.y;

    ImGui::SetNextWindowPos(
        ImVec2(screenW - panelW - 20.f, screenH * 0.12f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH),
                               ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
    ImGui::Begin("##panel_face_detection", nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddText(ImVec2(wp.x, wp.y + 8.f),
                IM_COL32(0, 230, 200, 220), "FACE DETECTION");
    dl->AddLine(ImVec2(wp.x, wp.y + 26.f),
                 ImVec2(wp.x + panelW, wp.y + 26.f),
                 IM_COL32(0, 200, 180, 80), 1.0f);

    ImGui::SetCursorPosY(38.f);
    ImGui::TextWrapped("Face detection is now controlled from the main panel.");

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// â”€â”€â”€ Sub-panel: Add Data (Face / Object) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
void HUDLayer::drawPanel_AddData(TranslationTaskManager* /*tm*/) {
    float panelW = 220.f;
    float panelH = 320.f;
    float screenW = ImGui::GetIO().DisplaySize.x;
    float screenH = ImGui::GetIO().DisplaySize.y;

    ImGui::SetNextWindowPos(
        ImVec2(screenW - panelW - 20.f, screenH * 0.12f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));

    ImGui::Begin("##panel_add_data", nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddText(ImVec2(wp.x, wp.y + 8.f),
                IM_COL32(0, 230, 200, 220), "ADD DATA");
    dl->AddLine(ImVec2(wp.x, wp.y + 26.f),
                 ImVec2(wp.x + panelW, wp.y + 26.f),
                 IM_COL32(0, 200, 180, 80), 1.0f);

    ImGui::SetCursorPosY(38.f);

    auto techButton = [&](const char* label, const char* id, bool isActive) -> bool {
        ImVec4 btnCol = ImVec4(0.f, 0.f, 0.f, 0.35f);
        ImVec4 hovCol = ImVec4(0.07f, 0.43f, 0.9f, 0.55f);
        ImVec4 actCol = ImVec4(0.1f, 0.55f, 1.f, 0.85f);
        if (isActive) {
            hovCol = ImVec4(0.05f, 0.35f, 0.85f, 0.75f);
            btnCol = ImVec4(0.08f, 0.45f, 1.0f, 0.90f);
            actCol = btnCol;
        }

        ImGui::PushStyleColor(ImGuiCol_Button, btnCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hovCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, actCol);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.95f, 1.f, 1.f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.f);
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.f, 0.8f, 0.7f, 0.3f));
        ImGui::SetNextItemWidth(-1.f);

        std::string full = std::string(label) + "##" + id;
        bool clicked = ImGui::Button(full.c_str(), ImVec2(-1.f, 32.f));

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);
        return clicked;
    };

    // Buttons (top)
    if (techButton("ADD FACE", "add_face", selectedAddDataIndex_ == 0)) {
        selectedAddDataIndex_ = 0;
        // Start enrollment state machine
        if (enrollState_ == EnrollState::IDLE || enrollState_ == EnrollState::DONE) {
            enrollState_ = EnrollState::WAITING_NAME;
            std::memset(enrollNameBuf_, 0, sizeof(enrollNameBuf_));
            enrollNameFocusSet_ = false;
            capturedFaces_.clear();
            statusOverride_.clear();
            std::cout << "[Enroll] Entered WAITING_NAME state\n";
        }
    }
    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("ADD OBJECT", "add_object", selectedAddDataIndex_ == 1)) {
        selectedAddDataIndex_ = 1;
        pendingAction_ = {HUDActionType::AddObject, -1};
    }
    ImGui::Spacing(); ImGui::Spacing();

    // ---- Enrollment UI ------------------------------------------------------
    static int s_enrMode = 0;
    static char s_enrName[64] = {};
    static int s_enrIdx = 0;
    static float s_enrProg = 0.f;

    ImGui::Separator();
    ImGui::Text("Enroll");
    ImGui::RadioButton("Face", &s_enrMode, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Object", &s_enrMode, 2);
    ImGui::SameLine();
    ImGui::RadioButton("Off", &s_enrMode, 0);

    if (s_enrMode != 0) {
        ImGui::InputText("Label", s_enrName, sizeof(s_enrName));
        ImGui::InputInt("Target #", &s_enrIdx);
        if (s_enrProg > 0.f && s_enrProg < 1.f)
            ImGui::ProgressBar(s_enrProg, ImVec2(-1.f, 0.f));
        if (techButton("CAPTURE", "cap", s_enrName[0] != '\0') &&
            s_enrName[0] != '\0') {
            pendingAction_.type = (s_enrMode == 1)
                                        ? HUDActionType::AddFace
                                        : HUDActionType::AddObject;
            pendingAction_.taskIdx = s_enrIdx;
            pendingAction_.text = s_enrName;
            s_enrProg = 0.01f;
        }
        ImGui::Spacing(); ImGui::Spacing();
    }

    // Face enrollment UI (below buttons)
    if (selectedAddDataIndex_ == 0) {
        if (enrollState_ == EnrollState::WAITING_NAME) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.f, 1.f, 0.f, 1.f),
                               "Enter person's name:");
            if (!enrollNameFocusSet_) {
                ImGui::SetKeyboardFocusHere(0);
                enrollNameFocusSet_ = true;
            }
            ImGui::SetNextItemWidth(200.f);
            if (ImGui::InputText("##enroll_name", enrollNameBuf_,
                                 sizeof(enrollNameBuf_),
                                 ImGuiInputTextFlags_EnterReturnsTrue)) {
                if (std::strlen(enrollNameBuf_) > 0) {
                    enrolledName_ = std::string(enrollNameBuf_);
                    enrollState_ = EnrollState::CAPTURING;
                    capturedFaces_.clear();
                    std::cout << "[Enroll] Name accepted: "
                              << enrolledName_ << "\n";
                }
            }
        } else if (enrollState_ == EnrollState::CAPTURING) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.f, 1.f, 0.f, 1.f),
                               "Capturing: %zu / 50 â€” keep face in frame",
                               capturedFaces_.size());
        } else if (enrollState_ == EnrollState::ENCODING) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.f, 0.8f, 0.f, 1.f),
                               "Encoding face for: %s ...",
                               enrolledName_.c_str());
        } else if (enrollState_ == EnrollState::DONE) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.f, 1.f, 0.f, 1.f),
                               "Identity saved: %s",
                               enrolledName_.c_str());
            ImGui::TextUnformatted(
                "Press Enter on 'Add Face' to enroll another person.");
        }
    } else if (selectedAddDataIndex_ == 1) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, kGreen);
        ImGui::TextUnformatted(
            "Mode: ADD OBJECT (select region on video and capture)");
        ImGui::PopStyleColor();
    }

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// â”€â”€â”€ Sub-panel: Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Styled: section separators, cyan headers, dim hint text, colored badges.
static const ImVec4 kSettingsBg     = ImVec4(0.12f, 0.12f, 0.16f, 1.0f);
static const ImVec4 kSectionAccent  = ImVec4(0.4f, 0.8f, 1.0f, 1.0f);
static const ImVec4 kDimText        = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
static const ImVec4 kActiveGreen    = ImVec4(0.2f, 1.0f, 0.4f, 1.0f);
static const ImVec4 kDisabledGrey   = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
static const ImVec4 kErrorRed       = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);

static void settingsSectionHeader(const char* title) {
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, kSectionAccent);
    ImGui::PushFont(ImGui::GetFont());  // bold via larger size if available
    ImGui::SetWindowFontScale(1.1f);
    ImGui::TextUnformatted(title);
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopFont();
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Separator, kSectionAccent);
    ImGui::Separator();
    ImGui::PopStyleColor();
    ImGui::Spacing();
}

void HUDLayer::drawPanel_Settings(TranslationTaskManager* /*tm*/) {
    float panelW = 220.f;
    float panelH = 320.f;
    float screenW = ImGui::GetIO().DisplaySize.x;
    float screenH = ImGui::GetIO().DisplaySize.y;

    ImGui::SetNextWindowPos(
        ImVec2(screenW - panelW - 20.f, screenH * 0.12f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));

    ImGui::Begin("##panel_settings", nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddText(ImVec2(wp.x, wp.y + 8.f),
                IM_COL32(0, 230, 200, 220), "SETTINGS");
    dl->AddLine(ImVec2(wp.x, wp.y + 26.f),
                 ImVec2(wp.x + panelW, wp.y + 26.f),
                 IM_COL32(0, 200, 180, 80), 1.0f);

    ImGui::SetCursorPosY(38.f);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, kSettingsBg);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.f);
    if (!ImGui::BeginChild("##settings_content", ImVec2(avail.x, avail.y - 50.f), true, ImGuiWindowFlags_NoScrollbar)) {
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        ImGui::End();
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(3);
        return;
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, 1.f));

    // â”€â”€â”€ 1. Detection Resolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    settingsSectionHeader("Detection Resolution");
    static const struct { const char* label; int w, h; } kRes[] = {
        {"320 x 240",   320,  240},
        {"640 x 480",   640,  480},
        {"960 x 540",   960,  540},
        {"1280 x 720", 1280, 720},
    };
    static int resIdx = 1;
    for (int i = 0; i < 4; ++i) {
        if (kRes[i].w == Config::detectionWidth && kRes[i].h == Config::detectionHeight) {
            resIdx = i;
            break;
        }
    }
    ImGui::SetNextItemWidth(-1.f);
    if (ImGui::BeginCombo("##det_res", kRes[resIdx].label)) {
        for (int i = 0; i < 4; ++i) {
            bool sel = (i == resIdx);
            if (ImGui::Selectable(kRes[i].label, sel)) {
                resIdx = i;
                Config::detectionWidth  = kRes[i].w;
                Config::detectionHeight = kRes[i].h;
                gResultsStore.setFeedInfo(
                    Config::CAPTURE_WIDTH, Config::CAPTURE_HEIGHT,
                    kRes[i].w, kRes[i].h);
            }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    {
        HUDSnapshot snap = gResultsStore.snapshot();
        ImGui::PushStyleColor(ImGuiCol_Text, kDimText);
        ImGui::Text("Capture: %d x %d  |  Detection: %d x %d",
                    snap.captureW, snap.captureH, snap.detectionW, snap.detectionH);
        ImGui::PopStyleColor();
    }

    // â”€â”€â”€ 2. Target Language â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    settingsSectionHeader("Target Language");
    static const struct { const char* label; const char* code; } kLangs[] = {
        {"English", "en"}, {"Japanese", "ja"}, {"French", "fr"}, {"German", "de"},
        {"Korean", "ko"}, {"Arabic", "ar"}, {"Chinese (Simplified)", "zh"},
    };
    static int langIdx = 0;
    for (int i = 0; i < 7; ++i) {
        if (Config::targetLanguage == kLangs[i].code) { langIdx = i; break; }
    }
    ImGui::SetNextItemWidth(-1.f);
    if (ImGui::BeginCombo("##tgt_lang", kLangs[langIdx].label)) {
        for (int i = 0; i < 7; ++i) {
            bool sel = (i == langIdx);
            if (ImGui::Selectable(kLangs[i].label, sel)) {
                langIdx = i;
                Config::targetLanguage = kLangs[i].code;
                gTranslationQueue.clear();
            }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    // â”€â”€â”€ 3. Modules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    settingsSectionHeader("Modules");
    auto drawModuleCheck = [](const char* label, bool* v) {
        ImGui::PushStyleColor(ImGuiCol_CheckMark, *v ? ImVec4(0.0f, 0.85f, 0.9f, 1.0f) : kDisabledGrey);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, 1.f));
        ImGui::Checkbox(label, v);
        ImGui::PopStyleColor(2);
    };
    drawModuleCheck("STT  Speech to Text", &Config::sttEnabled);
    drawModuleCheck("OCR  Camera to Text", &Config::ocrEnabled);
    drawModuleCheck("TTS  Speak Results",  &Config::ttsEnabled);

    // â”€â”€â”€ 4. TTS Volume â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    settingsSectionHeader("TTS Volume");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::SliderFloat("##vol", &Config::ttsVolume, 0.f, 1.f, "%.2f");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, kDimText);
    ImGui::Text("%.2f", Config::ttsVolume);
    ImGui::PopStyleColor();

    // â”€â”€â”€ 5. HUD Position â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    settingsSectionHeader("HUD Position");
    ImGui::Text("HUD Y: %+.2f", Config::hudPositionY);
    ImGui::PushStyleColor(ImGuiCol_Text, kDimText);
    ImGui::TextUnformatted("Ctrl+Up/Down to move HUD vertically");
    ImGui::PopStyleColor();

    // â”€â”€â”€ 6. GPU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    settingsSectionHeader("GPU");
    bool isCuda = (Config::gpuProvider.find("CUDA") != std::string::npos);
    ImVec4 badgeCol = isCuda ? kActiveGreen : ImVec4(0.95f, 0.75f, 0.05f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, badgeCol);
    ImGui::Text("Provider: %s", Config::gpuProvider.c_str());
    ImGui::PopStyleColor();

    // â”€â”€â”€ 7. Module Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    settingsSectionHeader("Module Status");
    auto statusRow = [&](const char* name, const std::string& status) {
        ImVec4 col = kDisabledGrey;
        const char* icon = "[ ]";
        if (status == "disabled" || status == "error") {
            icon = (status == "error") ? "[!]" : "[ ]";
            col = (status == "error") ? kErrorRed : kDisabledGrey;
        } else if (status == "ready" || status == "idle" || status == "listening" || status == "active") {
            icon = "[v]";
            col = kActiveGreen;
        } else if (status == "speaking" || status == "processing") {
            icon = "[~]";
            col = kSectionAccent;
        }
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        ImGui::Text("%s  %-4s : %s", icon, name, status.empty() ? "disabled" : status.c_str());
        ImGui::PopStyleColor();
    };
    HUDSnapshot snapB = gResultsStore.snapshot();
    statusRow("STT",  snapB.sttStatus.empty() ? "disabled" : snapB.sttStatus);
    statusRow("OCR",  snapB.ocrStatus.empty() ? "disabled" : snapB.ocrStatus);
    statusRow("TTS",  snapB.ttsStatus.empty() ? "disabled" : snapB.ttsStatus);
    statusRow("PLAY", snapB.ttsPlaying ? "active" : "idle");

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::EndChild();
    ImGui::Spacing(); ImGui::Spacing();

    // Back is handled via Left Arrow; no visible back button.

    ImGui::PopStyleColor(1);
    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Sub-panel: OCR Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawPanel_OCR(TranslationTaskManager* tm) {
    float panelW = 220.f;
    float panelH = 320.f;
    float screenW = ImGui::GetIO().DisplaySize.x;
    float screenH = ImGui::GetIO().DisplaySize.y;

    HUDSnapshot snap = gResultsStore.snapshot();
    const TranslationTask curTask =
        (tm ? tm->getCurrentTask() : TranslationTask::OCR_TO_TEXT);

    ImGui::SetNextWindowPos(
        ImVec2(screenW - panelW - 20.f, screenH * 0.12f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));

    ImGui::Begin("##panel_ocr", nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddText(ImVec2(wp.x, wp.y + 8.f),
                IM_COL32(0, 230, 200, 220), "OCR");
    dl->AddLine(ImVec2(wp.x, wp.y + 26.f),
                 ImVec2(wp.x + panelW, wp.y + 26.f),
                 IM_COL32(0, 200, 180, 80), 1.0f);

    ImGui::SetCursorPosY(38.f);

    auto techButton = [&](const char* label, const char* id, bool isActive) -> bool {
        ImVec4 btnCol = ImVec4(0.f, 0.f, 0.f, 0.35f);
        ImVec4 hovCol = ImVec4(0.07f, 0.43f, 0.9f, 0.55f);
        ImVec4 actCol = ImVec4(0.1f, 0.55f, 1.f, 0.85f);
        if (isActive) {
            hovCol = ImVec4(0.05f, 0.35f, 0.85f, 0.75f);
            btnCol = ImVec4(0.08f, 0.45f, 1.0f, 0.90f);
            actCol = btnCol;
        }

        ImGui::PushStyleColor(ImGuiCol_Button, btnCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hovCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, actCol);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.95f, 1.f, 1.f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.f);
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.f, 0.8f, 0.7f, 0.3f));
        ImGui::SetNextItemWidth(-1.f);

        std::string full = std::string(label) + "##" + id;
        bool clicked = ImGui::Button(full.c_str(), ImVec2(-1.f, 32.f));

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);
        return clicked;
    };

    // Live scan toggle
    bool liveOn = snap.ocrLiveOn;
    if (ImGui::Checkbox("Live Scan (auto)", &liveOn))
        gResultsStore.setOCRLiveOn(liveOn);
    ImGui::SameLine();
    ImGui::TextDisabled("(R)");

    ImGui::Spacing(); ImGui::Spacing();

    // Capture mode buttons (mutually exclusive visual highlight)
    bool cropMode = snap.ocrCropMode;
    if (techButton("Capture Now", "cap_now", !cropMode)) {
        gResultsStore.setOCRUseCrop(false);
        gResultsStore.setOCRCaptureRequested(true);
    }
    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("Click to Crop", "cap_crop", cropMode)) {
        gResultsStore.setOCRCropMode(!cropMode);
    }
    ImGui::Spacing(); ImGui::Spacing();

    // Confidence
    ImGui::PushStyleColor(ImGuiCol_Text,
                           ImVec4(0.f, 0.9f, 0.75f, 0.9f));
    ImGui::Text("Confidence: %.0f%%", snap.ocrConfidence);
    ImGui::PopStyleColor();
    ImVec4 confColor = snap.ocrConfidence > 60.f ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0.5f, 0, 1);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, confColor);
    ImGui::ProgressBar(snap.ocrConfidence / 100.f, ImVec2(-1, 10));
    ImGui::PopStyleColor();

    ImGui::Spacing(); ImGui::Spacing();

    // OCR -> Speech disabled; keep OCR as text-only.
    Config::ocrOutputToSpeech = false;
    ImGui::TextDisabled("Output: Text (Speech disabled)");

    ImGui::Spacing(); ImGui::Spacing();

    // Output mode (OCR -> Text)
    if (techButton("OCR -> Text", "ocr_to_text", curTask == TranslationTask::OCR_TO_TEXT)) {
        if (tm) tm->setTask(TranslationTask::OCR_TO_TEXT);
        pendingAction_ = {HUDActionType::SetTask, (int)TranslationTask::OCR_TO_TEXT};
    }
    ImGui::Spacing(); ImGui::Spacing();

    ImGui::Text("Detected Text:");
    ImGui::BeginChild("ocr_text_box", ImVec2(-1, 120), true,
                         ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::TextWrapped("%s",
                        snap.ocrOriginal.empty() ? "(none)" : snap.ocrOriginal.c_str());
    ImGui::EndChild();

    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("Copy", "ocr_copy", !snap.ocrOriginal.empty())) {
        ImGui::SetClipboardText(snap.ocrOriginal.c_str());
    }
    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("Clear", "ocr_clear", true)) {
        gResultsStore.setOCROriginal("");
        gResultsStore.setLastSpokenOcr("");
    }
    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("Save .txt", "ocr_save", !snap.ocrOriginal.empty())) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
        std::string path = "ocr_" + std::to_string(ms) + ".txt";
        std::ofstream f(path);
        if (f.is_open()) {
            f << snap.ocrOriginal;
            f.close();
        }
    }
    ImGui::Spacing(); ImGui::Spacing();

    // Back is handled via Left Arrow; no visible back button.

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Sub-panel: Sign Language Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawPanel_Sign(TranslationTaskManager* tm) {
    float panelW = 220.f;
    float panelH = 320.f;
    float screenW = ImGui::GetIO().DisplaySize.x;
    float screenH = ImGui::GetIO().DisplaySize.y;

    const TranslationTask curTask =
        (tm ? tm->getCurrentTask() : TranslationTask::NONE);

    ImGui::SetNextWindowPos(
        ImVec2(screenW - panelW - 20.f, screenH * 0.12f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));

    ImGui::Begin("##panel_sign", nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddText(ImVec2(wp.x, wp.y + 8.f),
                IM_COL32(0, 230, 200, 220), "SIGN LANGUAGE");
    dl->AddLine(ImVec2(wp.x, wp.y + 26.f),
                 ImVec2(wp.x + panelW, wp.y + 26.f),
                 IM_COL32(0, 200, 180, 80), 1.0f);

    ImGui::SetCursorPosY(38.f);

    auto techButton = [&](const char* label, const char* id, bool isActive) -> bool {
        ImVec4 btnCol = ImVec4(0.f, 0.f, 0.f, 0.35f);
        ImVec4 hovCol = ImVec4(0.07f, 0.43f, 0.9f, 0.55f);
        ImVec4 actCol = ImVec4(0.1f, 0.55f, 1.f, 0.85f);
        if (isActive) {
            hovCol = ImVec4(0.05f, 0.35f, 0.85f, 0.75f);
            btnCol = ImVec4(0.08f, 0.45f, 1.0f, 0.90f);
            actCol = btnCol;
        }

        ImGui::PushStyleColor(ImGuiCol_Button, btnCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hovCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, actCol);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.95f, 1.f, 1.f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.f);
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.f, 0.8f, 0.7f, 0.3f));
        ImGui::SetNextItemWidth(-1.f);

        std::string full = std::string(label) + "##" + id;
        bool clicked = ImGui::Button(full.c_str(), ImVec2(-1.f, 32.f));

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);
        return clicked;
    };

    const bool signToTextActive = (curTask == TranslationTask::SIGN_TO_TEXT);
    const bool signToSpeechActive = (curTask == TranslationTask::SIGN_TO_SPEECH);

    if (techButton("SIGN -> TEXT", "sign_text", signToTextActive)) {
        if (tm) tm->setTask(TranslationTask::SIGN_TO_TEXT);
        pendingAction_ = {HUDActionType::SetTask, (int)TranslationTask::SIGN_TO_TEXT};
    }
    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("SIGN -> SPEECH", "sign_speech", signToSpeechActive)) {
        if (tm) tm->setTask(TranslationTask::SIGN_TO_SPEECH);
        pendingAction_ = {HUDActionType::SetTask, (int)TranslationTask::SIGN_TO_SPEECH};
    }
    ImGui::Spacing(); ImGui::Spacing();

    std::string signWord = gResultsStore.getSignWord();
    float signConf = gResultsStore.getSignConfidence();
    if (!signWord.empty() && signConf >= 0.6f) {
        ImGui::Text("Sign: %s (%.0f%%)", signWord.c_str(), signConf * 100.0f);
    }

    // Back is handled via Left Arrow; no visible back button.

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Sub-panel: Speech Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawPanel_Speech(TranslationTaskManager* tm) {
    float panelW = 220.f;
    float panelH = 320.f;
    float screenW = ImGui::GetIO().DisplaySize.x;
    float screenH = ImGui::GetIO().DisplaySize.y;

    HUDSnapshot snap = gResultsStore.snapshot();
    const TranslationTask curTask =
        (tm ? tm->getCurrentTask() : TranslationTask::SPEECH_TO_TEXT);

    ImGui::SetNextWindowPos(
        ImVec2(screenW - panelW - 20.f, screenH * 0.12f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));

    ImGui::Begin("##panel_speech", nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddText(ImVec2(wp.x, wp.y + 8.f),
                IM_COL32(0, 230, 200, 220), "SPEECH");
    dl->AddLine(ImVec2(wp.x, wp.y + 26.f),
                 ImVec2(wp.x + panelW, wp.y + 26.f),
                 IM_COL32(0, 200, 180, 80), 1.0f);

    ImGui::SetCursorPosY(38.f);

    auto techButton = [&](const char* label, const char* id, bool isActive) -> bool {
        ImVec4 btnCol = ImVec4(0.f, 0.f, 0.f, 0.35f);
        ImVec4 hovCol = ImVec4(0.07f, 0.43f, 0.9f, 0.55f);
        ImVec4 actCol = ImVec4(0.1f, 0.55f, 1.f, 0.85f);
        if (isActive) {
            hovCol = ImVec4(0.05f, 0.35f, 0.85f, 0.75f);
            btnCol = ImVec4(0.08f, 0.45f, 1.0f, 0.90f);
            actCol = btnCol;
        }

        ImGui::PushStyleColor(ImGuiCol_Button, btnCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hovCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, actCol);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.95f, 1.f, 1.f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.f);
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.f, 0.8f, 0.7f, 0.3f));
        ImGui::SetNextItemWidth(-1.f);

        std::string full = std::string(label) + "##" + id;
        bool clicked = ImGui::Button(full.c_str(), ImVec2(-1.f, 32.f));

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);
        return clicked;
    };

    std::string engineLabel = "Engine: ";
    engineLabel +=
        (gSTTEngine.getEngine() == STTEngine::Engine::WHISPER)
            ? "Whisper.cpp"
            : "SAPI";
    ImGui::TextDisabled("%s", engineLabel.c_str());

    ImGui::Spacing(); ImGui::Spacing();

    if (!snap.sttActive) {
        if (techButton("START LISTENING", "stt_start", true)) {
            asyncSTT(7) /* non-blocking async STT */;
        }
        ImGui::Spacing(); ImGui::Spacing();
    } else {
        if (techButton("STOP LISTENING", "stt_stop", true)) {
            gSTTEngine.stopListening();
        }
        ImGui::Spacing(); ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text,
                               ImVec4(1.f, 0.4f, 0.0f, 1.0f));
        ImGui::Text("%s", snap.sttPartial.empty() ? "Recording..."
                                                     : snap.sttPartial.c_str());
        ImGui::PopStyleColor();
        ImGui::Spacing(); ImGui::Spacing();
    }

    ImGui::Text("Result:");
    ImGui::BeginChild("stt_result_box", ImVec2(-1, 100), true);
    ImGui::TextWrapped("%s",
                        snap.sttOriginal.empty() ? "(none)"
                                                 : snap.sttOriginal.c_str());
    ImGui::EndChild();

    ImGui::Spacing(); ImGui::Spacing();

    if (!snap.sttOriginal.empty()) {
        if (techButton("SPEAK RESULT", "stt_speak", true)) {
            pendingAction_ = {HUDActionType::SpeakText, -1,
                               snap.sttOriginal};
        }
        ImGui::Spacing(); ImGui::Spacing();
        if (techButton("COPY", "stt_copy", true)) {
            ImGui::SetClipboardText(snap.sttOriginal.c_str());
        }
        ImGui::Spacing(); ImGui::Spacing();
        if (techButton("CLEAR", "stt_clear", true)) {
            gResultsStore.setSTTOriginal("", "en");
        }
        ImGui::Spacing(); ImGui::Spacing();
    }

    ImGui::TextDisabled("Shortcut: L = start/stop");

    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("SPEECH -> TEXT",
                    "speech_to_text",
                    curTask == TranslationTask::SPEECH_TO_TEXT)) {
        if (tm) tm->setTask(TranslationTask::SPEECH_TO_TEXT);
        pendingAction_ = {HUDActionType::SetTask,
                            (int)TranslationTask::SPEECH_TO_TEXT};
    }
    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("TEXT -> SPEECH",
                    "text_to_speech",
                    curTask == TranslationTask::TEXT_TO_SPEECH)) {
        if (tm) tm->setTask(TranslationTask::TEXT_TO_SPEECH);
        pendingAction_ = {HUDActionType::SetTask,
                            (int)TranslationTask::TEXT_TO_SPEECH};
    }

    // Back is handled via Left Arrow; no visible back button.

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Sub-panel: Translation Hub Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawPanel_Translation(TranslationTaskManager* tm) {
    float panelW = 220.f;
    float panelH = 320.f;
    float screenW = ImGui::GetIO().DisplaySize.x;
    float screenH = ImGui::GetIO().DisplaySize.y;

    ImGui::SetNextWindowPos(
        ImVec2(screenW - panelW - 20.f, screenH * 0.12f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));

    ImGui::Begin("##panel_translation", nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBackground);
    // Prevent ImGui's default nav highlight/outline (avoids double highlight)
    ImGui::PushStyleColor(ImGuiCol_NavHighlight, ImVec4(0.f, 0.f, 0.f, 0.f));

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddText(ImVec2(wp.x, wp.y + 8.f),
                IM_COL32(0, 230, 200, 220), "TRANSLATION");
    dl->AddLine(ImVec2(wp.x, wp.y + 26.f),
                 ImVec2(wp.x + panelW, wp.y + 26.f),
                 IM_COL32(0, 200, 180, 80), 1.0f);

    ImGui::SetCursorPosY(38.f);

    TranslationTask curTask =
        tm ? tm->getCurrentTask() : TranslationTask::SCENE_TO_SPEECH;

    // When we just entered Translation Hub, sync selection from current task
    {
        static UIPanel s_lastPanel = UIPanel::Main;
        if (s_lastPanel != UIPanel::Translation && uiPanel_ == UIPanel::Translation) {
            switch (curTask) {
                case TranslationTask::OCR_TO_TEXT:
                case TranslationTask::OCR_TO_SPEECH:
                    selectedTranslationIndex_ = 0;
                    break;
                case TranslationTask::SPEECH_TO_TEXT:
                    selectedTranslationIndex_ = 1;
                    break;
                case TranslationTask::TEXT_TO_SPEECH:
                    selectedTranslationIndex_ = 2;
                    break;
                case TranslationTask::SIGN_TO_TEXT:
                case TranslationTask::SIGN_TO_SPEECH:
                    selectedTranslationIndex_ = 3;
                    break;
                default:
                    selectedTranslationIndex_ = 0;
                    break;
            }
        }
        s_lastPanel = uiPanel_;
    }

    const bool ocrActive =
        (curTask == TranslationTask::OCR_TO_TEXT ||
         curTask == TranslationTask::OCR_TO_SPEECH);
    const bool signActive =
        (curTask == TranslationTask::SIGN_TO_TEXT ||
         curTask == TranslationTask::SIGN_TO_SPEECH);

    auto techButton = [&](const char* label,
                           const char* id,
                           bool isActive,
                           bool isSelected) -> bool {
        // Active = filled; Selected = border only (no fill)
        const bool fill = isActive;
        ImVec4 btnCol = ImVec4(0.f, 0.f, 0.f, 0.35f);
        ImVec4 hovCol = ImVec4(0.07f, 0.43f, 0.9f, 0.55f);
        ImVec4 actCol = ImVec4(0.1f, 0.55f, 1.f, 0.85f);
        if (fill) {
            hovCol = ImVec4(0.05f, 0.35f, 0.85f, 0.75f);
            btnCol = ImVec4(0.08f, 0.45f, 1.0f, 0.90f);
            actCol = btnCol;
        }

        ImGui::PushStyleColor(ImGuiCol_Button, btnCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hovCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, actCol);
        ImGui::PushStyleColor(ImGuiCol_Text,
                               ImVec4(0.85f, 0.95f, 1.f, 1.f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.f);
        ImGui::PushStyleColor(ImGuiCol_Border,
                               ImVec4(0.f, 0.8f, 0.7f, 0.3f));
        ImGui::SetNextItemWidth(-1.f);

        std::string full = std::string(label) + "##" + id;
        bool clicked = ImGui::Button(full.c_str(), ImVec2(-1.f, 32.f));

        if (isSelected) {
            ImDrawList* wdl = ImGui::GetWindowDrawList();
            if (wdl) {
                ImVec2 a = ImGui::GetItemRectMin();
                ImVec2 b = ImGui::GetItemRectMax();
                wdl->AddRect(a, b, IM_COL32(0, 230, 200, 180), 3.f, 0, 2.0f);
            }
        }

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);
        return clicked;
    };

    // Buttons
    if (techButton("OCR", "ocr", ocrActive,
                    selectedTranslationIndex_ == 0)) {
        selectedTranslationIndex_ = 0;
        if (tm) tm->setTask(TranslationTask::OCR_TO_TEXT);
        pendingAction_ = {HUDActionType::SetTask,
                            (int)TranslationTask::OCR_TO_TEXT};
    }
    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("SPEECH -> TEXT",
                    "speech_to_text",
                    curTask == TranslationTask::SPEECH_TO_TEXT,
                    selectedTranslationIndex_ == 1)) {
        selectedTranslationIndex_ = 1;
        if (tm) tm->setTask(TranslationTask::SPEECH_TO_TEXT);
        pendingAction_ = {HUDActionType::SetTask,
                            (int)TranslationTask::SPEECH_TO_TEXT};
    }
    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("TEXT -> SPEECH",
                    "text_to_speech",
                    curTask == TranslationTask::TEXT_TO_SPEECH,
                    selectedTranslationIndex_ == 2)) {
        selectedTranslationIndex_ = 2;
        if (tm) tm->setTask(TranslationTask::TEXT_TO_SPEECH);
        pendingAction_ = {HUDActionType::SetTask,
                            (int)TranslationTask::TEXT_TO_SPEECH};
    }
    ImGui::Spacing(); ImGui::Spacing();

    if (techButton("SIGN",
                    "sign",
                    signActive,
                    selectedTranslationIndex_ == 3)) {
        selectedTranslationIndex_ = 3;
        if (tm) tm->setTask(TranslationTask::SIGN_TO_TEXT);
        pendingAction_ = {HUDActionType::SetTask,
                            (int)TranslationTask::SIGN_TO_TEXT};
    }
    ImGui::Spacing(); ImGui::Spacing();

    // Back is handled via Left Arrow; no visible back button.

    ImGui::PopStyleColor(); // ImGuiCol_NavHighlight
    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Right task panel Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawNodeIcon(ImDrawList* dl, const ImVec2& c, UIPanel panel, ImU32 col, float size) {
    if (!dl) return;
    const float w = 1.5f;
    const float h = size * 0.5f;
    const float kTwoPi = 6.28318530717958647692f;

    auto line = [&](float x1, float y1, float x2, float y2) {
        dl->AddLine(ImVec2(c.x + x1, c.y + y1), ImVec2(c.x + x2, c.y + y2), col, w);
    };

    switch (panel) {
        case UIPanel::ObjectDetection: {
            const float s = size * 0.42f;
            const float k = size * 0.18f;
            line(-s, -s, -s + k, -s); line(-s, -s, -s, -s + k);
            line(+s, -s, +s - k, -s); line(+s, -s, +s, -s + k);
            line(-s, +s, -s + k, +s); line(-s, +s, -s, +s - k);
            line(+s, +s, +s - k, +s); line(+s, +s, +s, +s - k);
            dl->AddCircle(ImVec2(c.x, c.y), 1.2f, col, 12, w);
            break;
        }
        case UIPanel::FaceDetection: {
            dl->AddCircle(c, size * 0.30f, col, 32, w);
            ImVec2 browL[] = {
                ImVec2(c.x - h * 0.55f, c.y - h * 0.15f),
                ImVec2(c.x - h * 0.25f, c.y - h * 0.28f),
                ImVec2(c.x - h * 0.02f, c.y - h * 0.18f),
            };
            ImVec2 browR[] = {
                ImVec2(c.x + h * 0.02f, c.y - h * 0.18f),
                ImVec2(c.x + h * 0.25f, c.y - h * 0.28f),
                ImVec2(c.x + h * 0.55f, c.y - h * 0.15f),
            };
            dl->AddPolyline(browL, 3, col, ImDrawFlags_None, w);
            dl->AddPolyline(browR, 3, col, ImDrawFlags_None, w);
            dl->AddCircle(ImVec2(c.x, c.y + h * 0.05f), 1.2f, col, 12, w);
            break;
        }
        case UIPanel::Translation: {
            const float bw = size * 0.55f;
            const float bh = size * 0.38f;
            ImVec2 a0 = ImVec2(c.x - bw * 0.60f, c.y - bh * 0.35f);
            ImVec2 a1 = ImVec2(c.x + bw * 0.40f, c.y + bh * 0.45f);
            ImVec2 b0 = ImVec2(c.x - bw * 0.35f, c.y - bh * 0.55f);
            ImVec2 b1 = ImVec2(c.x + bw * 0.65f, c.y + bh * 0.25f);
            dl->AddRect(a0, a1, col, 4.f, 0, w);
            dl->AddRect(b0, b1, col, 4.f, 0, w);
            line(-bw * 0.15f, +bh * 0.45f, -bw * 0.05f, +bh * 0.62f);
            line(+bw * 0.30f, +bh * 0.25f, +bw * 0.42f, +bh * 0.40f);
            break;
        }
        case UIPanel::VoiceAssist: {
            const float r = size * 0.18f;
            const float topY = -size * 0.18f;
            const float botY = +size * 0.18f;
            dl->AddCircle(ImVec2(c.x, c.y + topY), r, col, 24, w);
            dl->AddCircle(ImVec2(c.x, c.y + botY), r, col, 24, w);
            line(-r, topY, -r, botY);
            line(+r, topY, +r, botY);
            line(0.f, botY + r, 0.f, botY + r + size * 0.18f);
            ImVec2 base[] = {
                ImVec2(c.x - size * 0.22f, c.y + botY + r + size * 0.18f),
                ImVec2(c.x,                c.y + botY + r + size * 0.25f),
                ImVec2(c.x + size * 0.22f, c.y + botY + r + size * 0.18f),
            };
            dl->AddPolyline(base, 3, col, ImDrawFlags_None, w);
            break;
        }
        case UIPanel::AddData: {
            const float rr = size * 0.26f;
            const float y0 = -size * 0.18f;
            const float y1 = +size * 0.22f;
            dl->AddCircle(ImVec2(c.x, c.y + y0), rr, col, 24, w);
            dl->AddCircle(ImVec2(c.x, c.y + y1), rr, col, 24, w);
            line(-rr, y0, -rr, y1);
            line(+rr, y0, +rr, y1);
            line(-rr * 0.50f, 0.f, +rr * 0.50f, 0.f);
            line(0.f, -rr * 0.50f, 0.f, +rr * 0.50f);
            break;
        }
        case UIPanel::Settings: {
            const float rr = size * 0.28f;
            dl->AddCircle(c, rr, col, 32, w);
            for (int i = 0; i < 6; ++i) {
                const float ang = kTwoPi * (float)i / 6.f;
                const float x0 = std::cos(ang) * (rr + 2.f);
                const float y0 = std::sin(ang) * (rr + 2.f);
                const float x1 = std::cos(ang) * (rr + 7.f);
                const float y1 = std::sin(ang) * (rr + 7.f);
                line(x0, y0, x1, y1);
            }
            break;
        }
        default:
            break;
    }
}

void HUDLayer::drawSubOptionIcon(ImDrawList* dl, ImVec2 c, UIPanel panel, int optionIndex, ImU32 col, float size) {
    if (!dl) return;

    const float w = 1.5f;
    auto line = [&](float x1, float y1, float x2, float y2) {
        dl->AddLine(ImVec2(c.x + x1, c.y + y1), ImVec2(c.x + x2, c.y + y2), col, w);
    };
    auto backArrow = [&]() {
        const float h = size * 0.35f;
        dl->AddLine(ImVec2(c.x + h, c.y - h), ImVec2(c.x - h, c.y), col, w);
        dl->AddLine(ImVec2(c.x + h, c.y + h), ImVec2(c.x - h, c.y), col, w);
        dl->AddLine(ImVec2(c.x - h, c.y),     ImVec2(c.x + h, c.y), col, w);
    };

    auto mic = [&]() {
        const float r = size * 0.18f;
        dl->AddCircle(ImVec2(c.x, c.y - r * 0.7f), r, col, 20, w);
        line(-r, -r * 0.7f, -r, +r * 0.8f);
        line(+r, -r * 0.7f, +r, +r * 0.8f);
        line(-r, +r * 0.8f, +r, +r * 0.8f);
        line(0.f, +r * 0.8f, 0.f, +r * 1.7f);
        line(-r * 0.9f, +r * 1.7f, +r * 0.9f, +r * 1.7f);
    };

    auto speaker = [&]() {
        const float s = size * 0.22f;
        // body (trapezoid-ish)
        line(-s * 1.6f, -s, -s * 0.6f, -s);
        line(-s * 1.6f, +s, -s * 0.6f, +s);
        line(-s * 1.6f, -s, -s * 1.6f, +s);
        line(-s * 0.6f, -s, +s * 0.2f, -s * 1.4f);
        line(-s * 0.6f, +s, +s * 0.2f, +s * 1.4f);
        line(+s * 0.2f, -s * 1.4f, +s * 0.2f, +s * 1.4f);
        // waves
        dl->AddCircle(ImVec2(c.x + s * 0.8f, c.y), s * 1.05f, col, 18, w);
        dl->AddCircle(ImVec2(c.x + s * 0.8f, c.y), s * 1.65f, col, 18, w);
    };

    auto face = [&](bool plus) {
        const float r = size * 0.26f;
        dl->AddCircle(c, r, col, 28, w);
        dl->AddCircle(ImVec2(c.x - r * 0.35f, c.y - r * 0.15f), r * 0.08f, col, 10, w);
        dl->AddCircle(ImVec2(c.x + r * 0.35f, c.y - r * 0.15f), r * 0.08f, col, 10, w);
        if (plus) {
            const float p = r * 0.35f;
            line(0.f, +p * 0.2f, 0.f, +p * 1.0f);
            line(-p * 0.4f, +p * 0.6f, +p * 0.4f, +p * 0.6f);
        }
    };

    auto bboxCorners = [&]() {
        const float s = size * 0.40f;
        const float k = size * 0.18f;
        // TL
        line(-s, -s, -s + k, -s); line(-s, -s, -s, -s + k);
        // TR
        line(+s, -s, +s - k, -s); line(+s, -s, +s, -s + k);
        // BL
        line(-s, +s, -s + k, +s); line(-s, +s, -s, +s - k);
        // BR
        line(+s, +s, +s - k, +s); line(+s, +s, +s, +s - k);
    };

    auto hand = [&]() {
        const float s = size * 0.22f;
        // 4 fingers
        line(-s * 1.2f, -s * 1.4f, -s * 1.2f, +s * 1.2f);
        line(-s * 0.4f, -s * 1.6f, -s * 0.4f, +s * 1.25f);
        line(+s * 0.4f, -s * 1.6f, +s * 0.4f, +s * 1.25f);
        line(+s * 1.2f, -s * 1.4f, +s * 1.2f, +s * 1.2f);
        // thumb
        line(-s * 1.4f, +s * 0.6f, -s * 0.2f, +s * 1.6f);
    };

    auto sliders = [&]() {
        const float s = size * 0.30f;
        // three lines
        line(-s * 1.2f, -s, +s * 1.2f, -s);
        line(-s * 1.2f,  0.f, +s * 1.2f,  0.f);
        line(-s * 1.2f, +s, +s * 1.2f, +s);
        // knobs
        dl->AddCircle(ImVec2(c.x - s * 0.2f, c.y - s), s * 0.18f, col, 12, w);
        dl->AddCircle(ImVec2(c.x + s * 0.5f, c.y),      s * 0.18f, col, 12, w);
        dl->AddCircle(ImVec2(c.x - s * 0.6f, c.y + s),  s * 0.18f, col, 12, w);
    };

    auto eye = [&]() {
        // Approximate "eye" with circle + pupil
        const float r = size * 0.28f;
        dl->AddCircle(c, r, col, 24, w);
        dl->AddCircle(c, r * 0.18f, col, 12, w);
    };

    auto camera = [&]() {
        const float s = size * 0.28f;
        ImVec2 p[5] = {
            ImVec2(c.x - s * 1.4f, c.y - s),
            ImVec2(c.x + s * 1.4f, c.y - s),
            ImVec2(c.x + s * 1.4f, c.y + s),
            ImVec2(c.x - s * 1.4f, c.y + s),
            ImVec2(c.x - s * 1.4f, c.y - s),
        };
        dl->AddPolyline(p, 5, col, ImDrawFlags_None, w);
        dl->AddCircle(c, s * 0.55f, col, 18, w);
        // small top bump
        line(-s * 0.6f, -s, -s * 0.2f, -s * 1.35f);
        line(-s * 0.2f, -s * 1.35f, +s * 0.2f, -s * 1.35f);
        line(+s * 0.2f, -s * 1.35f, +s * 0.6f, -s);
    };

    auto ocrA = [&]() {
        const float s = size * 0.30f;
        ImVec2 p[5] = {
            ImVec2(c.x - s, c.y - s),
            ImVec2(c.x + s, c.y - s),
            ImVec2(c.x + s, c.y + s),
            ImVec2(c.x - s, c.y + s),
            ImVec2(c.x - s, c.y - s),
        };
        dl->AddPolyline(p, 5, col, ImDrawFlags_None, w);
        // "A"
        line(-s * 0.45f, +s * 0.55f, 0.f, -s * 0.55f);
        line(+s * 0.45f, +s * 0.55f, 0.f, -s * 0.55f);
        line(-s * 0.20f, +s * 0.05f, +s * 0.20f, +s * 0.05f);
    };

    auto cubePlus = [&]() {
        const float s = size * 0.22f;
        ImVec2 f[5] = {
            ImVec2(c.x - s, c.y - s),
            ImVec2(c.x + s, c.y - s),
            ImVec2(c.x + s, c.y + s),
            ImVec2(c.x - s, c.y + s),
            ImVec2(c.x - s, c.y - s),
        };
        ImVec2 b[5] = {
            ImVec2(c.x - s * 0.45f, c.y - s * 1.45f),
            ImVec2(c.x + s * 1.55f, c.y - s * 1.45f),
            ImVec2(c.x + s * 1.55f, c.y + s * 0.55f),
            ImVec2(c.x - s * 0.45f, c.y + s * 0.55f),
            ImVec2(c.x - s * 0.45f, c.y - s * 1.45f),
        };
        dl->AddPolyline(f, 5, col, ImDrawFlags_None, w);
        dl->AddPolyline(b, 5, col, ImDrawFlags_None, w);
        line(-s, -s, -s * 0.45f, -s * 1.45f);
        line(+s, -s, +s * 1.55f, -s * 1.45f);
        line(+s, +s, +s * 1.55f, +s * 0.55f);
        line(-s, +s, -s * 0.45f, +s * 0.55f);
        // plus (bottom)
        line(0.f, +s * 0.6f, 0.f, +s * 1.35f);
        line(-s * 0.35f, +s * 0.98f, +s * 0.35f, +s * 0.98f);
    };

    auto handWithT = [&]() {
        hand();
        // "T"
        const float s = size * 0.20f;
        line(-s * 0.6f, +s * 1.25f, +s * 0.6f, +s * 1.25f);
        line(0.f, +s * 1.25f, 0.f, +s * 2.0f);
    };

    auto handWithSpeaker = [&]() {
        hand();
        // small speaker to the right
        ImVec2 cc(c.x + size * 0.28f, c.y + size * 0.18f);
        const float s = size * 0.10f;
        dl->AddCircle(ImVec2(cc.x + s * 1.0f, cc.y), s * 1.3f, col, 16, w);
        dl->AddCircle(ImVec2(cc.x + s * 1.0f, cc.y), s * 0.8f, col, 16, w);
        dl->AddLine(ImVec2(cc.x - s * 1.4f, cc.y - s * 0.6f), ImVec2(cc.x - s * 0.2f, cc.y - s * 0.6f), col, w);
        dl->AddLine(ImVec2(cc.x - s * 1.4f, cc.y + s * 0.6f), ImVec2(cc.x - s * 0.2f, cc.y + s * 0.6f), col, w);
        dl->AddLine(ImVec2(cc.x - s * 1.4f, cc.y - s * 0.6f), ImVec2(cc.x - s * 1.4f, cc.y + s * 0.6f), col, w);
    };

    auto micWithAIDot = [&]() {
        mic();
        dl->AddCircle(ImVec2(c.x + size * 0.32f, c.y + size * 0.28f), size * 0.06f, col, 10, w);
    };

    // Panel-specific option icons (optionIndex includes BACK as last)
    switch (panel) {
        case UIPanel::ObjectDetection:
            if (optionIndex == 0) bboxCorners();
            else if (optionIndex == 1) sliders();     // CONFIG
            else backArrow();
            return;

        case UIPanel::FaceDetection:
            if (optionIndex == 0) face(false);        // SCAN
            else if (optionIndex == 1) face(true);    // ADD
            else backArrow();
            return;

        case UIPanel::Translation:
            if (optionIndex == 0) ocrA();             // OCR
            else if (optionIndex == 1) mic();         // S->T
            else if (optionIndex == 2) speaker();     // T->S
            else if (optionIndex == 3) hand();        // SIGN
            else backArrow();
            return;

        case UIPanel::Speech:
            if (optionIndex == 0) mic();              // VOICE
            else backArrow();
            return;

        case UIPanel::AddData:
            if (optionIndex == 0) face(true);         // FACE+
            else if (optionIndex == 1) cubePlus();    // OBJECT+
            else backArrow();
            return;

        case UIPanel::Settings:
            if (optionIndex == 0) sliders();          // CONFIG
            else backArrow();
            return;

        case UIPanel::OCR:
            if (optionIndex == 0) eye();              // LIVE
            else if (optionIndex == 1) camera();      // CAPTURE
            else backArrow();
            return;

        case UIPanel::Sign:
            if (optionIndex == 0) handWithT();        // S->T
            else if (optionIndex == 1) handWithSpeaker(); // S->S
            else backArrow();
            return;

        case UIPanel::VoiceAssist:
            if (optionIndex == 0) speaker();          // FEEDBACK
            else if (optionIndex == 1) micWithAIDot();// ASSIST
            else backArrow();
            return;

        default:
            dl->AddCircle(c, size * 0.08f, col, 10, w);
            return;
    }
}

void HUDLayer::drawMinimizedTaskStrip() {
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };

    ImGuiIO& io = ImGui::GetIO();
    const float screenW = io.DisplaySize.x;
    const float screenH = io.DisplaySize.y;
    if (screenW <= 0.f || screenH <= 0.f) return;

    const float panelW = (panelVisible_ && !m_minimizedMode) ? (screenW * PANEL_FRAC) : 0.f;
    const float videoW = screenW - panelW;
    const float stripRight = videoW - 10.f;
    ImGui::SetNextWindowPos(ImVec2(stripRight - 120.f, screenH * 0.15f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(120.f, screenH * 0.70f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    ImGui::Begin("##hud_strip", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse |
        ImGuiWindowFlags_NoBackground |
        ImGuiWindowFlags_NoNav);

    struct TaskNode { UIPanel panel; const char* label; };
    static const TaskNode nodes[kNodeCount] = {
        { UIPanel::ObjectDetection, "OBJECT"     },
        { UIPanel::FaceDetection,   "FACE"       },
        { UIPanel::Translation,     "TRANSL"     },
        { UIPanel::Speech,          "VOICE"      },
        { UIPanel::AddData,         "DATA"       },
        { UIPanel::Settings,        "SETTINGS"   },
    };

    if (m_stripSelectedIndex < 0) m_stripSelectedIndex = 0;
    if (m_stripSelectedIndex >= kNodeCount) m_stripSelectedIndex = kNodeCount - 1;

    ImDrawList* draw = ImGui::GetWindowDrawList();
    ImDrawList* fg   = ImGui::GetForegroundDrawList();
    const ImVec2 wp = ImGui::GetWindowPos();
    const float spacing = 72.0f;
    const float startY  = 40.0f;
    const float centerX = ImGui::GetWindowSize().x * 0.5f;

    for (int i = 0; i < kNodeCount; ++i) {
        // Screen-space center for this node (independent of ImGui cursor state)
        ImVec2 sc = ImVec2(wp.x + centerX, wp.y + startY + (float)i * spacing);

        bool isActivated = false;
        if (i == 0) isActivated = detectionActive_;
        else if (i == 1) isActivated = faceDetectionActive_;
        else isActivated = false;

        bool isHovered = (i == m_stripSelectedIndex);

        // Activation handled in handleInput() (draw-only)

        const float t = (float)ImGui::GetTime();

        if (!isHovered && !isActivated) {
            const float radius = 7.0f;
            draw->AddCircle(sc, radius, IM_COL32(255, 255, 255, 28), 32, 1.0f);
            const float pulse = 0.04f + 0.06f * (0.5f + 0.5f * std::sinf(t * 1.2f + (float)i));
            draw->AddCircleFilled(sc, radius, IM_COL32(255, 255, 255, (int)(pulse * 255.f)));
            m_nodeRadius[i] = radius;
        } else if (isHovered && !isActivated) {
            m_nodeRadius[i] = lerp(m_nodeRadius[i], 26.0f, 0.18f);
            draw->AddCircleFilled(sc, m_nodeRadius[i], IM_COL32(255, 255, 255, 26));
            draw->AddCircle(sc, m_nodeRadius[i], IM_COL32(255, 255, 255, 140), 32, 1.5f);

            if (!m_wasHovered[i]) m_hoverStartTime[i] = ImGui::GetTime();
            const float pingAge = (float)(ImGui::GetTime() - m_hoverStartTime[i]);
            if (pingAge < 0.5f) {
                const float pingRadius = m_nodeRadius[i] + pingAge * 60.0f;
                const float pingAlpha  = 1.0f - (pingAge / 0.5f);
                draw->AddCircle(sc, pingRadius, IM_COL32(255, 255, 255, (int)(pingAlpha * 100.f)), 32, 1.0f);
            }

            drawNodeIcon(draw, sc, nodes[i].panel, IM_COL32(255, 255, 255, 220), 20.0f);
            ImVec2 textSz = ImGui::CalcTextSize(nodes[i].label);
            ImVec2 labelPos(sc.x - textSz.x * 0.5f, sc.y + m_nodeRadius[i] + 5.0f);
            fg->AddText(labelPos, IM_COL32(220, 240, 255, 210), nodes[i].label);
        } else if (!isHovered && isActivated) {
            m_nodeRadius[i] = lerp(m_nodeRadius[i], 9.0f, 0.18f);
            const float glowAlpha = 0.5f + 0.35f * std::sinf(t * 2.5f);

            draw->AddCircleFilled(sc, m_nodeRadius[i], IM_COL32(26, 95, 255, 230));
            draw->AddCircle(sc, m_nodeRadius[i], IM_COL32(80, 160, 255, 180), 32, 1.0f);
            draw->AddCircle(sc, m_nodeRadius[i] + 4.0f, IM_COL32(30, 100, 255, (int)(glowAlpha * 120.f)), 32, 2.0f);

            const ImVec2 pip(sc.x + m_nodeRadius[i] * 0.65f, sc.y - m_nodeRadius[i] * 0.65f);
            draw->AddCircleFilled(pip, 2.5f, IM_COL32(125, 211, 255, 255));
        } else {
            m_nodeRadius[i] = lerp(m_nodeRadius[i], 26.0f, 0.18f);
            draw->AddCircleFilled(sc, m_nodeRadius[i], IM_COL32(20, 58, 204, 240));
            draw->AddCircle(sc, m_nodeRadius[i], IM_COL32(100, 180, 255, 216), 32, 1.5f);
            draw->AddCircle(sc, m_nodeRadius[i] + 3.0f, IM_COL32(30, 100, 255, 51), 32, 3.0f);
            draw->AddCircle(sc, m_nodeRadius[i] + 9.0f, IM_COL32(30, 100, 255, 80), 32, 1.0f);

            drawNodeIcon(draw, sc, nodes[i].panel, IM_COL32(255, 255, 255, 230), 20.0f);
            ImVec2 textSz = ImGui::CalcTextSize(nodes[i].label);
            ImVec2 labelPos(sc.x - textSz.x * 0.5f, sc.y + m_nodeRadius[i] + 5.0f);
            fg->AddText(labelPos, IM_COL32(144, 200, 255, 230), nodes[i].label);

            const ImVec2 pip(sc.x + m_nodeRadius[i] * 0.65f, sc.y - m_nodeRadius[i] * 0.65f);
            draw->AddCircleFilled(pip, 2.5f, IM_COL32(125, 211, 255, 255));
        }

        m_wasHovered[i] = isHovered;
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
}

void HUDLayer::drawMinimizedSubPanel() {
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };

    ImGuiIO& io = ImGui::GetIO();
    const float screenW = io.DisplaySize.x;
    const float screenH = io.DisplaySize.y;
    if (screenW <= 0.f || screenH <= 0.f) return;

    // These panels have NO second layer — render nothing.
    if (uiPanel_ == UIPanel::ObjectDetection) return;
    if (uiPanel_ == UIPanel::FaceDetection) return;
    if (uiPanel_ == UIPanel::AddData) return;
    if (uiPanel_ == UIPanel::Settings) return;

    ImGui::SetNextWindowPos(ImVec2(screenW - 200.f, screenH * 0.15f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(140.f, screenH * 0.70f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    ImGui::Begin("##hud_subpanel", nullptr,
        ImGuiWindowFlags_NoTitleBar  |
        ImGuiWindowFlags_NoResize    |
        ImGuiWindowFlags_NoMove      |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoBackground|
        ImGuiWindowFlags_NoNav);

    struct OptionNode { const char* label; int id; };
    OptionNode options[16] = {};
    int optionCount = 0;

    switch (uiPanel_) {
        case UIPanel::Translation:
            options[0] = {"OCR",    0};
            options[1] = {"S->T",   1};
            options[2] = {"T->S",   2};
            options[3] = {"SIGN",   3};
            optionCount = 4;
            break;
        case UIPanel::Speech:
            options[0] = {"S2T",   0};
            optionCount = 1;
            break;
        default:
            optionCount = 0;
            break;
    }

    if (optionCount <= 0) {
        ImGui::End();
        ImGui::PopStyleVar(2);
        return;
    }

    // BACK nodes are included directly in the option list for each panel.

    if (m_subpanelSelectedIndex < 0) m_subpanelSelectedIndex = 0;
    if (m_subpanelSelectedIndex >= optionCount) m_subpanelSelectedIndex = optionCount - 1;

    const float centerX = 70.f;
    const float spacing = 100.f;
    const float startY  = 92.f;
    const ImVec2 wp = ImGui::GetWindowPos();
    ImDrawList* dl = ImGui::GetForegroundDrawList();

    auto drawBackArrow = [&](const ImVec2& c, ImU32 col, float size) {
        const float w = 2.0f;
        const float s = size;
        dl->AddLine(ImVec2(c.x + s * 0.35f, c.y), ImVec2(c.x - s * 0.35f, c.y), col, w);
        dl->AddLine(ImVec2(c.x - s * 0.35f, c.y), ImVec2(c.x - s * 0.10f, c.y - s * 0.22f), col, w);
        dl->AddLine(ImVec2(c.x - s * 0.35f, c.y), ImVec2(c.x - s * 0.10f, c.y + s * 0.22f), col, w);
    };

    const TranslationTask curTask =
        taskMgr_ ? taskMgr_->getCurrentTask() : TranslationTask::NONE;

    // Activation is handled in handleInput() for reliability (Right Arrow).

    for (int i = 0; i < optionCount; i++) {
        const bool isSelected = (i == m_subpanelSelectedIndex);

        // "Active" (blue) state should represent real task activation only
        bool isActive = false;
    if (uiPanel_ == UIPanel::FaceDetection && options[i].id == 0)
            isActive = faceDetectionActive_;
        else if (uiPanel_ == UIPanel::Translation && options[i].id == 0)
            isActive = (curTask == TranslationTask::OCR_TO_TEXT || curTask == TranslationTask::OCR_TO_SPEECH);
        else if (uiPanel_ == UIPanel::Translation && options[i].id == 1)
            isActive = (curTask == TranslationTask::SPEECH_TO_TEXT);
        else if (uiPanel_ == UIPanel::Translation && options[i].id == 2)
            isActive = (curTask == TranslationTask::TEXT_TO_SPEECH);
        else if (uiPanel_ == UIPanel::Translation && options[i].id == 3)
            isActive = (curTask == TranslationTask::SIGN_TO_TEXT || curTask == TranslationTask::SIGN_TO_SPEECH);

        if (options[i].id == -1) {
            isActive = false;
        }

        ImVec2 sc = ImVec2(wp.x + centerX, wp.y + startY + (float)i * spacing);

        const float targetR = isSelected ? 30.0f : 12.0f;
        m_subpanelRadius[i] = lerp(m_subpanelRadius[i], targetR, 0.18f);

        const float t = (float)ImGui::GetTime();

        if (!isSelected && !isActive) {
            // Inactive: small subtle pulse
            dl->AddCircle(sc, m_subpanelRadius[i], IM_COL32(255, 255, 255, 40), 32, 1.0f);
            const float pulse = 0.04f + 0.06f * (0.5f + 0.5f * std::sinf(t * 1.2f + (float)i));
            dl->AddCircleFilled(sc, m_subpanelRadius[i], IM_COL32(255, 255, 255, (int)(pulse * 255.f)));
            drawSubOptionIcon(dl, sc, uiPanel_, i, IM_COL32(255, 255, 255, 140), 16.0f);
            continue;
        }

        if (!isSelected && isActive) {
            // Active but not selected: blue small node
            const float r = m_subpanelRadius[i];
            const float glowAlpha = 0.5f + 0.35f * std::sinf(t * 2.5f + (float)i);
            dl->AddCircleFilled(sc, r, IM_COL32(26, 95, 255, 230));
            dl->AddCircle(sc, r, IM_COL32(80, 160, 255, 180), 32, 1.0f);
            dl->AddCircle(sc, r + 4.0f, IM_COL32(30, 100, 255, (int)(glowAlpha * 120.f)), 32, 2.0f);
            drawSubOptionIcon(dl, sc, uiPanel_, i, IM_COL32(255, 255, 255, 230), 18.0f);
            continue;
        }

        // Selected (expanded): show label + icon + outline
        if (isActive) {
            dl->AddCircleFilled(sc, m_subpanelRadius[i], IM_COL32(20, 58, 204, 240));
            dl->AddCircle(sc, m_subpanelRadius[i], IM_COL32(100, 180, 255, 216), 32, 1.5f);
            dl->AddCircle(sc, m_subpanelRadius[i] + 3.0f, IM_COL32(30, 100, 255, 51), 32, 3.0f);
        } else {
            dl->AddCircleFilled(sc, m_subpanelRadius[i], IM_COL32(255, 255, 255, 20));
            dl->AddCircle(sc, m_subpanelRadius[i], IM_COL32(255, 255, 255, 180), 32, 1.5f);
        }

        drawSubOptionIcon(dl, sc, uiPanel_, i, IM_COL32(255, 255, 255, 230), 22.0f);

        ImVec2 textSz   = ImGui::CalcTextSize(options[i].label);
        ImVec2 labelPos = ImVec2(sc.x - textSz.x * 0.5f, sc.y + m_subpanelRadius[i] + 6.0f);
        dl->AddText(labelPos, IM_COL32(220, 240, 255, 220), options[i].label);

    }

    ImGui::End();
    ImGui::PopStyleVar(2);
}

void HUDLayer::drawMinimizedAddDataPanel() {
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };

    ImGuiIO& io = ImGui::GetIO();
    const float screenW = io.DisplaySize.x;
    const float screenH = io.DisplaySize.y;
    if (screenW <= 0.f || screenH <= 0.f) return;

    ImGui::SetNextWindowPos(ImVec2(screenW - 200.f, screenH * 0.15f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(140.f, screenH * 0.70f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);

    ImGui::Begin("##adddata_panel", nullptr,
        ImGuiWindowFlags_NoTitleBar  |
        ImGuiWindowFlags_NoResize    |
        ImGuiWindowFlags_NoMove      |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();

    static const char* options[2] = {"FACE", "OBJECT"};

    for (int i = 0; i < 2; i++) {
        bool isSelected = (i == m_subpanelSelectedIndex);
        ImVec2 sc = ImVec2(wp.x + 70.f,
                           wp.y + 60.f + i * 100.f);

        float r = isSelected ? 30.f : 12.f;
        m_subpanelRadius[i] = lerp(m_subpanelRadius[i], r, 0.18f);

        if (!isSelected) {
            dl->AddCircle(sc, m_subpanelRadius[i],
                IM_COL32(255,255,255,40), 32, 1.0f);
        } else {
            dl->AddCircleFilled(sc, m_subpanelRadius[i],
                IM_COL32(255,255,255,20));
            dl->AddCircle(sc, m_subpanelRadius[i],
                IM_COL32(255,255,255,180), 32, 1.5f);

            drawSubOptionIcon(dl, sc, UIPanel::AddData,
                i, IM_COL32(255,255,255,230), 22.0f);

            ImVec2 textSz = ImGui::CalcTextSize(options[i]);
            ImVec2 labelPos = ImVec2(sc.x - textSz.x*0.5f,
                                     sc.y + m_subpanelRadius[i]+6.f);
            dl->AddText(labelPos,
                IM_COL32(220,240,255,220), options[i]);
        }
    }

    ImGui::End();
}

void HUDLayer::drawMinimizedSettingsPanel() {
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };

    ImGuiIO& io = ImGui::GetIO();
    const float screenW = io.DisplaySize.x;
    const float screenH = io.DisplaySize.y;
    if (screenW <= 0.f || screenH <= 0.f) return;

    ImGui::SetNextWindowPos(ImVec2(screenW - 200.f, screenH * 0.15f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(140.f, screenH * 0.70f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);

    ImGui::Begin("##settings_panel", nullptr,
        ImGuiWindowFlags_NoTitleBar  |
        ImGuiWindowFlags_NoResize    |
        ImGuiWindowFlags_NoMove      |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();

    static const char* options[1] = {"CONFIG"};

    for (int i = 0; i < 1; i++) {
        bool isSelected = (i == m_subpanelSelectedIndex);
        ImVec2 sc = ImVec2(wp.x + 70.f,
                           wp.y + 60.f + i * 100.f);

        float r = isSelected ? 30.f : 12.f;
        m_subpanelRadius[i] = lerp(m_subpanelRadius[i], r, 0.18f);

        if (!isSelected) {
            dl->AddCircle(sc, m_subpanelRadius[i],
                IM_COL32(255,255,255,40), 32, 1.0f);
        } else {
            dl->AddCircleFilled(sc, m_subpanelRadius[i],
                IM_COL32(255,255,255,20));
            dl->AddCircle(sc, m_subpanelRadius[i],
                IM_COL32(255,255,255,180), 32, 1.5f);

            drawSubOptionIcon(dl, sc, UIPanel::Settings,
                i, IM_COL32(255,255,255,230), 22.0f);

            ImVec2 textSz = ImGui::CalcTextSize(options[i]);
            ImVec2 labelPos = ImVec2(sc.x - textSz.x*0.5f,
                                     sc.y + m_subpanelRadius[i]+6.f);
            dl->AddText(labelPos,
                IM_COL32(220,240,255,220), options[i]);
        }
    }

    ImGui::End();
}

void HUDLayer::drawTaskPanel(float panelX, float panelW, float h,
                              TranslationTaskManager* tm)
{
    // In minimized HUD mode, never draw the legacy right panel container.
    // Sub-panels are rendered as floating overlay windows instead.
    if (m_minimizedMode) { return; }

    const float yOff = Config::hudPositionY * h * 0.5f;
    ImGui::SetNextWindowPos ({panelX, yOff});
    ImGui::SetNextWindowSize({panelW, h});
    const bool isMainPanel = (uiPanel_ == UIPanel::Main);
    ImGui::SetNextWindowBgAlpha(isMainPanel ? 0.96f : 0.0f);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, isMainPanel ? kPanelBg : ImVec4(0.f, 0.f, 0.f, 0.f));
    ImGui::PushStyleColor(ImGuiCol_Border,   isMainPanel ? kCyanDim : ImVec4(0.f, 0.f, 0.f, 0.f));
    ImGui::PushStyleVar  (ImGuiStyleVar_WindowPadding, ImVec2(8.f, 10.f));
    ImGui::PushStyleVar  (ImGuiStyleVar_ItemSpacing,   ImVec2(4.f, 6.f));

    ImGui::Begin("##tasks", nullptr,
        ImGuiWindowFlags_NoTitleBar    |
        ImGuiWindowFlags_NoResize      |
        ImGuiWindowFlags_NoMove        |
        ImGuiWindowFlags_NoScrollbar   |
        ImGuiWindowFlags_NoCollapse    |
        ImGuiWindowFlags_NoNavInputs   |  // we handle Enter/arrows ourselves in handleInput
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Dispatch to correct sub-panel
    switch (uiPanel_) {
        case UIPanel::Main:             drawPanel_Main(tm);             break;
        case UIPanel::ObjectDetection:  drawPanel_ObjectDetection(tm);  break;
        case UIPanel::FaceDetection:    drawPanel_FaceDetection(tm);    break;
        case UIPanel::OCR:              drawPanel_OCR(tm);              break;
        case UIPanel::Sign:             drawPanel_Sign(tm);             break;
        case UIPanel::Speech:           drawPanel_Speech(tm);           break;
        case UIPanel::Translation:      drawPanel_Translation(tm);      break;
        case UIPanel::VoiceAssist:      drawPanel_VoiceAssist(tm);      break;
        case UIPanel::AddData:          drawPanel_AddData(tm);          break;
        case UIPanel::Settings:         drawPanel_Settings(tm);         break;
    }

    
    
    // -- Model status (real state) in bottom-right corner --
    ImGui::Separator();
    bool yOk = registry_ && registry_->isYOLOLoaded();
    bool oOk = registry_ && registry_->isOCRLoaded();
    bool fOk = registry_ && registry_->isFaceLoaded();

    bool ortKnown = registry_ && registry_->isYOLOLoaded();
    bool ortCuda  = registry_ && registry_->isYOLOCuda();
    const char* ortLabel = ortKnown ? (ortCuda ? "ORT:CUDA" : "ORT:CPU") : "ORT:N/A";

    const char* yText = yOk ? "YOLO:ON"  : "YOLO:OFF";
    const char* oText = oOk ? " OCR:ON"  : " OCR:OFF";
    const char* fText = fOk ? " FACE:ON" : " FACE:OFF";

    ImVec2 ySz   = ImGui::CalcTextSize(yText);
    ImVec2 oSz   = ImGui::CalcTextSize(oText);
    ImVec2 fSz   = ImGui::CalcTextSize(fText);
    ImVec2 ortSz = ImGui::CalcTextSize(ortLabel);

    float spacing    = 8.f;
    float totalTopW  = ySz.x + oSz.x + fSz.x + spacing * 2.f;
    ImVec2 winSize   = ImGui::GetWindowSize();
    float lineH      = ImGui::GetTextLineHeightWithSpacing();

    // Reserve three lines (YOLO/OCR/FACE, ORT backend, FPS)
    float startY = winSize.y - lineH * 3.f - 6.f;
    if (startY < ImGui::GetCursorPosY())
        startY = ImGui::GetCursorPosY();

    // First line: YOLO / OCR / FACE aligned to bottom-right of panel
    ImGui::SetCursorPosY(startY);
    float startX = winSize.x - totalTopW - 10.f;
    if (startX < 0.f) startX = 0.f;
    ImGui::SetCursorPosX(startX);

    ImGui::PushStyleColor(ImGuiCol_Text,
        yOk ? ImVec4(0.3f,0.9f,0.3f,1.f) : ImVec4(0.9f,0.3f,0.3f,1.f));
    ImGui::TextUnformatted(yText);
    ImGui::PopStyleColor();

    ImGui::SameLine(0.f, spacing);
    ImGui::PushStyleColor(ImGuiCol_Text,
        oOk ? ImVec4(0.3f,0.9f,0.3f,1.f) : ImVec4(0.9f,0.3f,0.3f,1.f));
    ImGui::TextUnformatted(oText);
    ImGui::PopStyleColor();

    ImGui::SameLine(0.f, spacing);
    ImGui::PushStyleColor(ImGuiCol_Text,
        fOk ? ImVec4(0.3f,0.9f,0.3f,1.f) : ImVec4(0.9f,0.3f,0.3f,1.f));
    ImGui::TextUnformatted(fText);
    ImGui::PopStyleColor();

    // Second line: ORT backend label directly below, right-aligned
    ImGui::SetCursorPosY(startY + lineH);
    float ortX = winSize.x - ortSz.x - 10.f;
    if (ortX < 0.f) ortX = 0.f;
    ImGui::SetCursorPosX(ortX);

    ImVec4 ortCol;
    if (!ortKnown)
        ortCol = ImVec4(0.7f,0.7f,0.7f,1.f);
    else if (ortCuda)
        ortCol = ImVec4(0.3f,0.9f,0.3f,1.f);
    else
        ortCol = ImVec4(0.95f,0.55f,0.05f,1.f); // CPU = orange

    ImGui::PushStyleColor(ImGuiCol_Text, ortCol);
    ImGui::TextUnformatted(ortLabel);
    ImGui::PopStyleColor();

    // Third line: live FPS under ORT label
    ImGui::SetCursorPosY(startY + lineH * 2.f);
    char fpsBuf[32];
    std::snprintf(fpsBuf, sizeof(fpsBuf), "FPS: %.1f", lastFps_);
    ImVec2 fpsSz = ImGui::CalcTextSize(fpsBuf);
    float fpsX = winSize.x - fpsSz.x - 10.f;
    if (fpsX < 0.f) fpsX = 0.f;
    ImGui::SetCursorPosX(fpsX);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f,0.9f,1.0f,1.f));
    ImGui::TextUnformatted(fpsBuf);
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Video background (left 85%) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawVideoBackground(float videoW, float h, GLuint texID) {
    ImGui::SetNextWindowPos({0.f, 0.f});
    ImGui::SetNextWindowSize({videoW, h});
    ImGui::SetNextWindowBgAlpha(1.f);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 1.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));

    ImGui::Begin("##video", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoInputs |
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    if (texID != 0) {
        ImGui::Image((ImTextureID)(uintptr_t)texID, ImVec2(videoW, h));
    } else {
        ImGui::SetCursorPos({videoW * 0.5f - 50.f, h * 0.5f - 10.f});
        ImGui::PushStyleColor(ImGuiCol_Text, kRed);
        ImGui::TextUnformatted("[ NO SIGNAL ]");
        ImGui::PopStyleColor();
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Detection boxes and face rectangles drawn on video Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawOverlay(float videoW, float h,
                            const std::vector<std::vector<float>>& boxes,
                            const std::vector<cv::Rect>& faces)
{
    // Simple temporal smoothing: keep last non-empty detections for a short
    // time window so bounding boxes do not flicker on/off every other frame.
    static std::vector<std::vector<float>> sLastBoxes;
    static std::vector<cv::Rect>           sLastFaces;
    static double                          sLastBoxTime  = 0.0;
    static double                          sLastFaceTime = 0.0;
    const double now = ImGui::GetTime();
    const double kHoldBoxesSeconds = 0.20; // keep object boxes ~200ms
    const double kHoldFacesSeconds = 0.35; // keep face boxes long enough to avoid flicker, but still responsive

    std::vector<std::vector<float>> boxesToDraw = boxes;
    std::vector<cv::Rect>           facesToDraw = faces;

    if (!boxes.empty()) {
        sLastBoxes  = boxes;
        sLastBoxTime = now;
    } else if ((now - sLastBoxTime) < kHoldBoxesSeconds) {
        boxesToDraw = sLastBoxes;
    }
    if (!faces.empty()) {
        sLastFaces   = faces;
        sLastFaceTime = now;
    } else if ((now - sLastFaceTime) < kHoldFacesSeconds) {
        facesToDraw = sLastFaces;
    }

    // Draw on top of all HUD windows so boxes are visible over the video.
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const float yOff = Config::hudPositionY * h * 0.5f;

    HUDSnapshot snap = gResultsStore.snapshot();

    // YOLO bounding boxes:
    // [x1_norm, y1_norm, x2_norm, y2_norm, classId, confidence]
    int detIdx = 0;
    for (const auto& b : boxesToDraw) {
        if (b.size() < 4) {
            ++detIdx;
            continue;
        }
        float x1 = b[0] * videoW, y1 = b[1] * h + yOff;
        float x2 = b[2] * videoW, y2 = b[3] * h + yOff;
        // Thicker rectangle for better visibility (objects = blue)
        dl->AddRect({x1, y1}, {x2, y2}, IM_COL32(0, 160, 255, 255), 0.0f, 0, 4.0f);

        if (b.size() >= 6) {
            int   clsId = static_cast<int>(b[4]);
            float conf  = b[5];

            const char* cocoName = nullptr;
            static std::string tmp;
            if (registry_) {
                const auto& names = registry_->getYOLOClassNames();
                if (clsId >= 0 && clsId < static_cast<int>(names.size()))
                    cocoName = names[clsId].c_str();
            }

            // Prefer custom label from ResultsStore if present for this detection index
            std::string labelText;
            if (detIdx < (int)snap.boxes.size() && !snap.boxes[detIdx].label.empty()) {
                labelText = snap.boxes[detIdx].label;
            } else if (cocoName && *cocoName) {
                labelText = cocoName;
            } else {
                labelText.clear();
            }

            char lbl[64];
            if (!labelText.empty()) {
                std::snprintf(lbl, sizeof(lbl), "%s %.0f%%", labelText.c_str(), conf * 100.f);
            } else {
                std::snprintf(lbl, sizeof(lbl), "%.0f%%", conf * 100.f);
            }

            // Slightly larger, high-contrast label with background
            ImVec2 textPos(x1 + 6.f, y1 + 4.f);
            ImVec2 textSize = ImGui::CalcTextSize(lbl);
            dl->AddRectFilled(
                ImVec2(textPos.x - 3.f, textPos.y - 2.f),
                ImVec2(textPos.x + textSize.x + 3.f, textPos.y + textSize.y + 2.f),
                IM_COL32(0, 0, 0, 180));
            dl->AddText(textPos, IM_COL32(0, 255, 255, 255), lbl);
        }
        ++detIdx;
    }

    // Face rectangles (pixel coords in original frame; re-scale to display)
    int faceIdx = 0;
    for (const auto& r : facesToDraw) {
        if (lastWidth_ <= 0 || lastHeight_ <= 0) break;
        float sx = videoW / (float)lastWidth_;
        float sy = h      / (float)lastHeight_;
        float x1 = r.x * sx, y1 = r.y * sy + yOff;
        float x2 = (r.x + r.width) * sx, y2 = (r.y + r.height) * sy + yOff;
        dl->AddRect({x1, y1}, {x2, y2}, IM_COL32(50, 230, 100, 255), 0.f, 0, 4.f);
        std::string faceLabel = "Unknown";
        if (faceIdx < (int)snap.faces.size() && !snap.faces[faceIdx].name.empty())
            faceLabel = snap.faces[faceIdx].name;

        ImVec2 textPos(x1 + 6.f, y1 + 4.f);
        ImVec2 textSize = ImGui::CalcTextSize(faceLabel.c_str());
        dl->AddRectFilled(
            ImVec2(textPos.x - 3.f, textPos.y - 2.f),
            ImVec2(textPos.x + textSize.x + 3.f, textPos.y + textSize.y + 2.f),
            IM_COL32(0, 0, 0, 180));
        dl->AddText(textPos, IM_COL32(120, 255, 160, 255), faceLabel.c_str());
        ++faceIdx;
    }
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Header bar Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::drawHeader(float w, float h,
                           const std::string& statusLine,
                           const std::string& hintLine,
                           float confidencePercent,
                           float fps)
{
    const float yOff = Config::hudPositionY * h * 0.5f;
    ImGui::SetNextWindowPos ({0.f, yOff});
    const float headerW = m_minimizedMode ? w : (w * (1.f - PANEL_FRAC));
    ImGui::SetNextWindowSize({headerW, HEADER_H});
    ImGui::SetNextWindowBgAlpha(0.88f);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, kHeaderBg);
    ImGui::PushStyleColor(ImGuiCol_Border,   kCyanDim);
    ImGui::PushStyleVar  (ImGuiStyleVar_WindowPadding, ImVec2(10.f, 6.f));

    ImGui::Begin("##header", nullptr,
        ImGuiWindowFlags_NoTitleBar   |
        ImGuiWindowFlags_NoResize     |
        ImGuiWindowFlags_NoMove       |
        ImGuiWindowFlags_NoScrollbar  |
        ImGuiWindowFlags_NoInputs     |
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    // Left: title
    ImGui::PushStyleColor(ImGuiCol_Text, kCyan);
    ImGui::TextUnformatted("SmartGlassesHUD");
    ImGui::PopStyleColor();

    // Middle: hint / output
    ImGui::SameLine(180.f);
    ImGui::PushStyleColor(ImGuiCol_Text, hintLine.empty() ? kCyanDim : kGreen);
    ImGui::TextUnformatted(hintLine.empty() ? "Waiting..." : hintLine.c_str());
    ImGui::PopStyleColor();

    // Confidence bar
    if (confidencePercent > 0.f) {
        ImGui::SameLine(500.f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              confidencePercent > 60.f ? kGreen : kOrange);
        char buf[16]; std::snprintf(buf, sizeof(buf), "%.0f%%", confidencePercent);
        ImGui::ProgressBar(confidencePercent / 100.f, {90.f, 14.f}, buf);
        ImGui::PopStyleColor();
    }

    // Right: FPS + status (HUD override wins if set)
    const std::string& statusSrc = !statusOverride_.empty() ? statusOverride_ : statusLine;
    char fpsBuf[64];
    std::snprintf(fpsBuf, sizeof(fpsBuf), "FPS: %.0f   %s", fps, statusSrc.c_str());
    float txtW = ImGui::CalcTextSize(fpsBuf).x;

    const char* hudMode = m_minimizedMode ? "HUD: MIN" : "HUD: ORIG";
    ImVec2 hudModeSz = ImGui::CalcTextSize(hudMode);
    const float pad = 12.f;
    const float gap = 14.f;
    const float startX = ImGui::GetWindowWidth() - (hudModeSz.x + gap + txtW) - pad;
    ImGui::SameLine(startX > 0.f ? startX : 0.f);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.f, 1.f, 0.78f, 0.5f));
    ImGui::TextUnformatted(hudMode);
    ImGui::PopStyleColor();

    ImGui::SameLine(0.f, gap);
    ImGui::PushStyleColor(ImGuiCol_Text,
        statusSrc.find("[LIVE]") != std::string::npos ? kGreen : kRed);
    ImGui::TextUnformatted(fpsBuf);
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
}

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Main draw Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void HUDLayer::draw(float displayWidth, float displayHeight,
                    const cv::Mat& frame,
                    const std::string& statusLine,
                    const std::string& hintLine,
                    float confidencePercent,
                    const std::vector<std::vector<float>>& detectionBoxes,
                    const std::vector<cv::Rect>& faceRects,
                    float fps,
                    TranslationTaskManager* taskMgr)
{
    if (!taskMgr && !taskMgr_) {
        return;
    }
    if (taskMgr) {
        taskMgr_ = taskMgr;
        lastKnownTask_ = taskMgr->getCurrentTask();
    } else if (taskMgr_) {
        lastKnownTask_ = taskMgr_->getCurrentTask();
    }

    // Detect transitions into TEXT_TO_SPEECH so we can request focus for the
    // TTS input box exactly once on entry (and never every frame).
    static TranslationTask prevTask = TranslationTask::NONE;
    if (lastKnownTask_ == TranslationTask::TEXT_TO_SPEECH &&
        prevTask != TranslationTask::TEXT_TO_SPEECH) {
        ttsInputFocusRequested_ = true;
        ttsOverlayJustOpened_   = true;
    }
    prevTask = lastKnownTask_;

    handleInput();

    // In minimized mode we never reserve the legacy right panel area.
    const bool showRightPanel = (panelVisible_ && !m_minimizedMode);
    const float panelW  = showRightPanel ? (displayWidth * PANEL_FRAC) : 0.f;
    const float videoW  = displayWidth - panelW;
    const float h       = displayHeight;

    // Cache latest FPS for bottom-right status block
    lastFps_ = fps;

    // 1. Upload frame to GPU (BGR->RGB fix happens here)
    GLuint texID = frame.empty() ? 0 : uploadFrame(frame);

    // 2. Video background (left 85%)
    drawVideoBackground(videoW, h, texID);

    // 3. Detection overlays drawn on top of video
    drawOverlay(videoW, h, detectionBoxes, faceRects);

    // 3b. Face enrollment capture (runs every frame, independent of keypress)
    updateEnrollment(frame, faceRects);

    bool showOcrOverlays = (lastKnownTask_ == TranslationTask::OCR_TO_TEXT || lastKnownTask_ == TranslationTask::OCR_TO_SPEECH);
    if (showOcrOverlays) {
        gResultsStore.tickOCRTimers(ImGui::GetIO().DeltaTime);
        HUDSnapshot snapV = gResultsStore.snapshot();
        drawOCRRegionOverlays(0.f, 0.f, videoW, h, snapV.captureW, snapV.captureH, taskMgr);
    }

    // 4. Header bar
    drawHeader(displayWidth, h, statusLine, hintLine, confidencePercent, fps);

    // 5. Legacy rectangular panel only in non-minimized mode
    if (!m_minimizedMode) {
        if (panelVisible_) {
            drawTaskPanel(videoW, panelW, h, taskMgr);
            drawLayerIndicator(displayWidth);
        }
    }

    // â”€â”€ Lower video feed overlays (STT, OCR, TTS output) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // All output in lower video area, not in task panel or header.
    {
        const float vW   = videoW;  // full width when panel hidden (Right toggle)
        const float vH   = displayHeight;
        const float yOff = Config::hudPositionY * displayHeight * 0.5f;
        TranslationTask cur = taskMgr ? taskMgr->getCurrentTask() : TranslationTask::NONE;

        bool showStt = (cur == TranslationTask::SPEECH_TO_TEXT);
        bool showOcr = (cur == TranslationTask::OCR_TO_TEXT || cur == TranslationTask::OCR_TO_SPEECH);
        bool showSign = (cur == TranslationTask::SIGN_TO_TEXT || cur == TranslationTask::SIGN_TO_SPEECH);
        bool showTts = (cur == TranslationTask::TEXT_TO_SPEECH);

        if (showOcr) {
            drawOCROverlay(vW, vH, yOff);
        }
        if (showSign) {
            drawSignOutputIcons(vW, vH, yOff);
        }
        if (showStt || showTts) drawSTTTTSInputOverlay(vW, vH, yOff, taskMgr);

        drawTTSIndicator();
        drawResolutionOverlay(vW, vH);
    }

    // Draw minimized UI last so it always receives input.
    if (m_minimizedMode && m_stripVisible) {
        // Panels with NO second layer should keep showing the main strip.
        if (uiPanel_ == UIPanel::Main ||
            uiPanel_ == UIPanel::ObjectDetection ||
            uiPanel_ == UIPanel::FaceDetection)
            drawMinimizedTaskStrip();
        else if (uiPanel_ == UIPanel::AddData)
            drawMinimizedAddDataPanel();
        else if (uiPanel_ == UIPanel::Settings)
            drawMinimizedSettingsPanel();
        else
            drawMinimizedSubPanel();
    }
}


// --- Sub-panel: Voice Assist ------------------------------------------------
void HUDLayer::drawPanel_VoiceAssist(TranslationTaskManager* /*tm*/) {
    float panelW = 220.f;
    float panelH = 320.f;
    float screenW = ImGui::GetIO().DisplaySize.x;
    float screenH = ImGui::GetIO().DisplaySize.y;

    ImGui::SetNextWindowPos(
        ImVec2(screenW - panelW - 20.f, screenH * 0.12f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH),
                               ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 12.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
    ImGui::Begin("##panel_voice_assist", nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddText(ImVec2(wp.x, wp.y + 8.f),
                IM_COL32(0, 230, 200, 220), "VOICE ASSIST");
    dl->AddLine(ImVec2(wp.x, wp.y + 26.f),
                 ImVec2(wp.x + panelW, wp.y + 26.f),
                 IM_COL32(0, 200, 180, 80), 1.0f);

    ImGui::SetCursorPosY(38.f);

    auto techButton = [&](const char* label,
                           const char* id,
                           bool isActive,
                           bool isSelected) -> bool {
        const bool fill = (isActive || isSelected);
        ImVec4 btnCol = ImVec4(0.f, 0.f, 0.f, 0.35f);
        ImVec4 hovCol = ImVec4(0.07f, 0.43f, 0.9f, 0.55f);
        ImVec4 actCol = ImVec4(0.1f, 0.55f, 1.f, 0.85f);
        if (fill) {
            hovCol = ImVec4(0.05f, 0.35f, 0.85f, 0.75f);
            btnCol = ImVec4(0.08f, 0.45f, 1.0f, 0.90f);
            actCol = btnCol;
        }

        ImGui::PushStyleColor(ImGuiCol_Button, btnCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hovCol);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, actCol);
        ImGui::PushStyleColor(ImGuiCol_Text,
                               ImVec4(0.85f, 0.95f, 1.f, 1.f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.f);
        ImGui::PushStyleColor(ImGuiCol_Border,
                               ImVec4(0.f, 0.8f, 0.7f, 0.3f));

        ImGui::SetNextItemWidth(-1.f);
        std::string full = std::string(label) + "##" + id;
        bool clicked = ImGui::Button(full.c_str(), ImVec2(-1.f, 32.f));

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);
        return clicked;
    };

    auto backButton = [&]() -> bool {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.2f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                               ImVec4(0.6f, 0.1f, 0.1f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_Text,
                               ImVec4(0.6f, 0.75f, 0.8f, 0.85f));
        bool clicked =
            ImGui::Button("<- BACK##back", ImVec2(-1.f, 28.f));
        ImGui::PopStyleColor(3);
        return clicked;
    };

    if (!gResultsStore.snapshot().sapiAvailable) {
        ImGui::PushStyleColor(ImGuiCol_Text, kRed);
        ImGui::TextWrapped(
            "SAPI not available. Install Windows Speech Platform or enable Windows TTS voices in Settings.");
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // Option 0: Voice Feedback
    const bool vfActive = voiceFeedback_;
    if (techButton(vfActive ? "VOICE FEEDBACK: ON" : "VOICE FEEDBACK: OFF",
                    "vf",
                    vfActive,
                    selectedVoiceAssistIndex_ == 0)) {
        selectedVoiceAssistIndex_ = 0;
        voiceFeedback_ = !voiceFeedback_;
        Config::TTS_ENABLED = voiceFeedback_;
        pendingAction_.type = HUDActionType::SetVoiceFeedback;
    }
    ImGui::Spacing();
    ImGui::Spacing();

    // Option 1: Voice Assist toggle
    if (techButton("VOICE ASSIST",
                    "va",
                    voiceAssistActive_,
                    selectedVoiceAssistIndex_ == 1)) {
        selectedVoiceAssistIndex_ = 1;
        bool newState = !voiceAssistActive_;
        voiceAssistActive_ = newState;
        pendingAction_ =
            {newState ? HUDActionType::StartVoiceAssist
                       : HUDActionType::StopVoiceAssist,
             -1};
    }
    ImGui::Spacing();
    ImGui::Spacing();

    // Status line (cyan tinted)
    ImGui::PushStyleColor(ImGuiCol_Text,
                           ImVec4(0.f, 0.9f, 0.75f, 0.9f));
    ImGui::Text(voiceAssistActive_ ? "ACTIVE - speaking every 4s"
                                    : "INACTIVE");
    ImGui::PopStyleColor();

    ImGui::SetCursorPosY(panelH - 44.f);
    if (backButton()) {
        uiPanel_ = UIPanel::Main;
        m_minimizedMode = true;
    }

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}



// -------------------------------------------------------
// HUDLayer::drawFaces â€” renders face boxes, landmarks,
// confidence, pose label onto the OpenCV frame
// -------------------------------------------------------
void HUDLayer::drawFaces(cv::Mat& frame,
                         const std::vector<FaceDetector::FaceResult>& faces)
{
    // Optional identity matching if encoder + store are available
    std::vector<hud::Identity> identities;
    if (faceEncoder_ && identityStore_) {
        identities = identityStore_->GetAll();
    }

    for (const auto& face : faces) {
        // --- Bounding Box (green) ---
        cv::rectangle(frame, face.box, cv::Scalar(0, 255, 0), 2);

        // --- Label: identity name if matched, else FACE #N / UNKNOWN ---
        std::string label = face.tag;
        if (!identities.empty() && faceEncoder_) {
            cv::Rect safe = face.box & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safe.width > 0 && safe.height > 0) {
                cv::Mat crop;
                cv::resize(frame(safe), crop, {112, 112});
                hud::Embedding emb = faceEncoder_->Encode(crop);
                if (!emb.empty()) {
                    float bestSim = 0.0f;
                    std::string bestName = "UNKNOWN";
                    for (const auto& ident : identities) {
                        float sim = hud::FaceEncoder::CosineSimilarity(emb, ident.embedding);
                        if (sim > bestSim) {
                            bestSim = sim;
                            bestName = ident.name;
                        }
                    }
                    if (bestSim >= 0.60f)
                        label = bestName;
                }
            }
        }
        if (label.empty())
            label = "UNKNOWN";

        cv::Size textSz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, nullptr);
        cv::Rect labelBg(face.box.x, face.box.y - textSz.height - 8,
                         textSz.width + 6, textSz.height + 8);
        labelBg &= cv::Rect(0, 0, frame.cols, frame.rows);
        cv::rectangle(frame, labelBg, cv::Scalar(0, 180, 0), cv::FILLED);
        cv::putText(frame, label,
                    {face.box.x + 3, face.box.y - 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        // --- Pose Label (white, below box) ---
        cv::putText(frame, face.poseLabel,
                    {face.box.x, face.box.y + face.box.height + 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        // --- Landmarks: eyes=cyan, nose=yellow, mouth=orange ---
        for (int k = 0; k < 5; k++) {
            cv::Scalar color = (k < 2) ? cv::Scalar(255,255,0)
                             : (k == 2) ? cv::Scalar(0,255,255)
                             : cv::Scalar(0,165,255);
            cv::circle(frame,
                       cv::Point(static_cast<int>(face.landmarks[k].x),
                                 static_cast<int>(face.landmarks[k].y)),
                       3, color, cv::FILLED, cv::LINE_AA);
        }
    }

    // --- Face Count (bottom-left) ---
    std::string status = "FACES: " + std::to_string(faces.size());
    cv::putText(frame, status, {10, frame.rows - 10},
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0,255,0), 2, cv::LINE_AA);
}

// â”€â”€â”€ OCR region overlays (Google Lens style): boxes, hover, click, floating text â”€
void HUDLayer::drawOCRRegionOverlays(float vX, float vY, float vW, float vH, int captureW, int captureH,
                                     TranslationTaskManager* tm) {
    if (!taskMgr_ && !tm) return;
    HUDSnapshot snap = gResultsStore.snapshot();
    if (snap.captureW <= 0 || snap.captureH <= 0) return;
    if (vW <= 0.0f || vH <= 0.0f) return;

    const float scaleX = vW / static_cast<float>(captureW);
    const float scaleY = vH / static_cast<float>(captureH);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 mouse = ImGui::GetMousePos();
    const bool ocrClick = ImGui::IsMouseClicked(0);

    for (size_t i = 0; i < snap.ocrRegions.size(); ++i) {
        const OCRRegion& r = snap.ocrRegions[i];
        ImVec2 tl(vX + r.bbox.x * scaleX, vY + r.bbox.y * scaleY);
        ImVec2 br(vX + (r.bbox.x + r.bbox.width) * scaleX,
                  vY + (r.bbox.y + r.bbox.height) * scaleY);

        bool isHovered = (mouse.x >= tl.x && mouse.x <= br.x &&
                          mouse.y >= tl.y && mouse.y <= br.y);
        bool isSelected = (snap.selectedOCRRegion >= 0 &&
                           static_cast<size_t>(snap.selectedOCRRegion) == i) || r.selected;

        dl->AddRect(tl, br, IM_COL32(0, 200, 255, 180), 0.0f, 0, 2.0f);
        dl->AddRectFilled(tl, br, IM_COL32(0, 200, 255, 20));

        if (isHovered) {
            dl->AddRectFilled(tl, br, IM_COL32(0, 200, 255, 60));
        }
        if (isSelected) {
            dl->AddRect(tl, br, IM_COL32(0, 255, 100, 255), 0.0f, 0, 2.5f);
        }

        // Label only when hovered or selected
        if (isHovered || isSelected) {
            std::string label = r.text.empty() ? getRegionLabel(r.bbox) : r.text;
            ImVec2 labelSize = ImGui::CalcTextSize(label.c_str());
            ImVec2 labelBgTL = ImVec2(tl.x, tl.y - labelSize.y - 4.f);
            ImVec2 labelBgBR = ImVec2(tl.x + labelSize.x + 8.f, tl.y);
            dl->AddRectFilled(labelBgTL, labelBgBR, IM_COL32(0, 200, 255, 200));
            dl->AddText(ImVec2(labelBgTL.x + 4.f, labelBgTL.y + 2.f),
                        IM_COL32(0, 0, 0, 255), label.c_str());
        }
    }

    if (ocrClick && mouse.x >= vX && mouse.x <= vX + vW && mouse.y >= vY && mouse.y <= vY + vH) {
        int fx = static_cast<int>((mouse.x - vX) / scaleX);
        int fy = static_cast<int>((mouse.y - vY) / scaleY);
        pendingAction_.type = HUDActionType::CaptureOCR;
        pendingAction_.taskIdx = -1;
        pendingAction_.ocrClickX = fx;
        pendingAction_.ocrClickY = fy;
    }

    if (snap.selectedOCRRegion >= 0 && snap.ocrOverlayTimer > 0.f && !snap.selectedOCRText.empty()) {
        if (static_cast<size_t>(snap.selectedOCRRegion) < snap.ocrRegions.size()) {
            const OCRRegion& sel = snap.ocrRegions[static_cast<size_t>(snap.selectedOCRRegion)];
            ImVec2 tl(vX + sel.bbox.x * scaleX, vY + sel.bbox.y * scaleY);
            float textW = ImGui::CalcTextSize(snap.selectedOCRText.c_str()).x + 16.f;
            float textH = 24.f;
            ImVec2 textPos(tl.x, tl.y - 30.f);
            if (textPos.y < vY) textPos.y = tl.y + sel.bbox.height * scaleY + 4.f;
            if (textPos.x + textW > vX + vW) textPos.x = vX + vW - textW;
            if (textPos.x < vX) textPos.x = vX;
            dl->AddRectFilled(textPos, ImVec2(textPos.x + textW, textPos.y + textH),
                              IM_COL32(0, 0, 0, 190), 4.f);
            dl->AddText(ImVec2(textPos.x + 8.f, textPos.y + 4.f), IM_COL32(255, 255, 255, 255),
                        snap.selectedOCRText.c_str());
        }
    }
}

// â”€â”€â”€ Task [10]: New overlay methods â€” read from gResultsStore â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

void HUDLayer::drawOCRCaptureButton(float vW, float vH, float yOff, TranslationTaskManager* /*tm*/) {
    const float btnSize = 48.f;
    const float margin = 16.f;
    const float ocrBoxH = 72.f;
    float px = vW - btnSize - margin;
    float py = vH - ocrBoxH - margin - btnSize - 8.f + yOff;

    ImGui::SetNextWindowPos(ImVec2(px, py), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(btnSize, btnSize));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.4f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, btnSize * 0.5f);

    if (ImGui::Begin("##ocr_capture_btn", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse)) {
        ImVec2 winMin = ImGui::GetWindowPos();
        ImVec2 winSize = ImGui::GetWindowSize();
        ImVec2 center(winMin.x + winSize.x * 0.5f, winMin.y + winSize.y * 0.5f);
        const float radius = 18.f;

        ImGui::InvisibleButton("##ocr_cap_btn", winSize);
        bool hover = ImGui::IsItemHovered();
        bool clicked = ImGui::IsItemClicked();
        if (clicked) {
            pendingAction_.type = HUDActionType::CaptureOCR;
            pendingAction_.taskIdx = -1;
        }

        static double s_ocrFlashUntil = 0.0;
        if (clicked) s_ocrFlashUntil = ImGui::GetTime() + 0.3;
        bool flash = (ImGui::GetTime() < s_ocrFlashUntil);
        HUDSnapshot snapBtn = gResultsStore.snapshot();

        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImU32 fillCol = IM_COL32(64, 64, 64, 230);
        ImU32 iconCol = IM_COL32(140, 140, 140, 255);
        if (flash) {
            fillCol = IM_COL32(0, 200, 255, 200);
            iconCol = IM_COL32(255, 255, 255, 255);
        } else if (hover) {
            iconCol = IM_COL32(255, 255, 255, 255);
        }
        dl->AddCircleFilled(center, radius, fillCol);
        dl->AddCircle(center, radius, iconCol, 0, 2.f);

        float cx = center.x, cy = center.y;
        float bw = 4.f, bh = 10.f;
        ImVec2 bodyMin(cx - bw * 0.5f, cy - bh * 0.5f);
        ImVec2 bodyMax(cx + bw * 0.5f, cy + bh * 0.5f);
        dl->AddRectFilled(bodyMin, bodyMax, iconCol, 2.f);
        dl->AddLine(ImVec2(cx - 6.f, cy + bh * 0.5f), ImVec2(cx + 6.f, cy + bh * 0.5f), iconCol, 1.5f);
        dl->AddLine(ImVec2(cx, cy + bh * 0.5f), ImVec2(cx, cy + 10.f), iconCol, 1.5f);
        dl->AddRectFilled(ImVec2(cx - 4.f, cy + 10.f), ImVec2(cx + 4.f, cy + 12.f), iconCol);

        if (hover)
            ImGui::SetTooltip("%s", snapBtn.ocrImageCaptured ? "Tap to read aloud" : "Tap to capture image");
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

void HUDLayer::drawOCROverlay(float vW, float vH, float yOff) {
    HUDSnapshot snap = gResultsStore.snapshot();

    // Bottom-right overlay: 60% width, 52px height
    const float overlayW = vW * 0.60f;
    const float overlayH = 52.f;
    const float overlayX = vW - overlayW - 8.f;
    const float overlayY = vH - overlayH - 8.f + yOff;

    ImGui::SetNextWindowPos(ImVec2(overlayX, overlayY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(overlayW, overlayH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.95f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.12f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.f, 0.85f, 1.f, 1.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.f, 8.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("##ocr_overlay", nullptr, flags)) {
        // Capture button (left)
        const float btnSize = 36.f;
        ImGui::InvisibleButton("##ocr_cap_btn", ImVec2(btnSize, btnSize));
        bool capClicked = ImGui::IsItemClicked();
        bool capHover = ImGui::IsItemHovered();
        if (capClicked) {
            pendingAction_.type = HUDActionType::CaptureOCR;
            pendingAction_.taskIdx = -1;
        }
        if (capHover)
            ImGui::SetTooltip("%s", snap.ocrImageCaptured ? "Tap to read aloud" : "Tap to capture image");
        ImVec2 rmin = ImGui::GetItemRectMin();
        ImVec2 rmax = ImGui::GetItemRectMax();
        ImVec2 center((rmin.x + rmax.x) * 0.5f, (rmin.y + rmax.y) * 0.5f);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImU32 fillCol = capHover ? IM_COL32(0, 200, 255, 150) : IM_COL32(64, 64, 64, 230);
        dl->AddCircleFilled(center, 14.f, fillCol);
        dl->AddCircle(center, 14.f, IM_COL32(140, 140, 140, 255), 0, 1.5f);
        float cx = center.x, cy = center.y;
        dl->AddRectFilled(ImVec2(cx - 3.f, cy - 5.f), ImVec2(cx + 3.f, cy + 5.f), IM_COL32(255, 255, 255, 255), 2.f);
        dl->AddLine(ImVec2(cx - 5.f, cy + 5.f), ImVec2(cx + 5.f, cy + 5.f), IM_COL32(255, 255, 255, 255), 1.5f);
        dl->AddLine(ImVec2(cx, cy + 5.f), ImVec2(cx, cy + 9.f), IM_COL32(255, 255, 255, 255), 1.5f);
        dl->AddRectFilled(ImVec2(cx - 3.f, cy + 9.f), ImVec2(cx + 3.f, cy + 11.f), IM_COL32(255, 255, 255, 255));

        ImGui::SameLine(0.f, 12.f);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (btnSize - ImGui::GetTextLineHeight()) * 0.5f);

        std::string displayText = snap.ocrOriginal.empty() ? snap.selectedOCRText : snap.ocrOriginal;
        if (!displayText.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, kCyan);
            ImGui::TextUnformatted("OCR:");
            ImGui::PopStyleColor();
            ImGui::SameLine(0.f, 8.f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, 1.f));
            ImGui::TextUnformatted(displayText.c_str());
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.55f, 1.f));
            ImGui::TextUnformatted(snap.ocrImageCaptured ? "Tap to read aloud" : "Tap to capture image");
            ImGui::PopStyleColor();
        }

        // T/S output mode icons (right side)
        ImGui::SameLine(overlayW - 70.f);
        bool textMode = !Config::ocrOutputToSpeech;
        ImGui::PushStyleColor(ImGuiCol_Button, textMode ? ImVec4(0.f, 0.45f, 0.55f, 1.f) : ImVec4(0.08f, 0.14f, 0.20f, 1.f));
        if (ImGui::Button("T", ImVec2(28.f, 0.f))) {
            Config::ocrOutputToSpeech = false;
            pendingAction_ = {HUDActionType::SetOCROutputText, -1};
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Output to Text");
        ImGui::PopStyleColor();
        ImGui::SameLine(0.f, 4.f);
        ImGui::PushStyleColor(ImGuiCol_Button, !textMode ? ImVec4(0.f, 0.45f, 0.55f, 1.f) : ImVec4(0.08f, 0.14f, 0.20f, 1.f));
        if (ImGui::Button("S", ImVec2(28.f, 0.f))) {
            Config::ocrOutputToSpeech = true;
            pendingAction_ = {HUDActionType::SetOCROutputSpeech, -1};
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Output to Speech");
        ImGui::PopStyleColor();
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

void HUDLayer::drawOCROutputIcons(float vW, float vH, float yOff) {
    const float iconSize = 28.f;
    const float gap = 6.f;
    const float ocrBoxH = 72.f;
    const float margin = 16.f;
    float py = vH - ocrBoxH - margin - iconSize - 8.f + yOff;
    float px = 12.f;

    ImGui::SetNextWindowPos(ImVec2(px, py), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(iconSize * 2.f + gap, iconSize));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, 0.f));

    if (ImGui::Begin("##ocr_output_icons", nullptr,
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 winMin = ImGui::GetWindowPos();

        bool textMode = !Config::ocrOutputToSpeech;
        ImVec2 tPos(winMin.x + 4.f, winMin.y + 4.f);
        ImVec2 tSize(iconSize - 8.f, iconSize - 8.f);
        ImGui::SetCursorScreenPos(tPos);
        ImGui::InvisibleButton("##ocr_t", tSize);
        if (ImGui::IsItemClicked()) {
            Config::ocrOutputToSpeech = false;
            pendingAction_ = {HUDActionType::SetOCROutputText, -1};
        }
        ImU32 tCol = textMode ? IM_COL32(0, 200, 255, 255) : IM_COL32(120, 120, 120, 200);
        dl->AddText(ImVec2(tPos.x + 6.f, tPos.y + 2.f), tCol, "T");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Output to Text");

        ImVec2 sPos(winMin.x + iconSize + gap + 4.f, winMin.y + 4.f);
        ImVec2 sSize(iconSize - 8.f, iconSize - 8.f);
        ImGui::SetCursorScreenPos(sPos);
        ImGui::InvisibleButton("##ocr_s", sSize);
        if (ImGui::IsItemClicked()) {
            Config::ocrOutputToSpeech = true;
            pendingAction_ = {HUDActionType::SetOCROutputSpeech, -1};
        }
        ImU32 sCol = !textMode ? IM_COL32(0, 200, 255, 255) : IM_COL32(120, 120, 120, 200);
        float cx = sPos.x + sSize.x * 0.5f, cy = sPos.y + sSize.y * 0.5f;
        dl->AddCircleFilled(ImVec2(cx - 3.f, cy), 3.f, sCol);
        dl->AddTriangleFilled(ImVec2(cx, cy - 4.f), ImVec2(cx, cy + 4.f), ImVec2(cx + 6.f, cy), sCol);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Output to Speech");
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

void HUDLayer::drawSignOutputIcons(float vW, float vH, float yOff) {
    const float iconSize = 28.f;
    const float gap = 6.f;
    const float margin = 16.f;
    float py = vH - margin - iconSize - 8.f + yOff;
    float px = 12.f;

    ImGui::SetNextWindowPos(ImVec2(px, py), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(iconSize * 2.f + gap, iconSize));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, 0.f));

    if (ImGui::Begin("##sign_output_icons", nullptr,
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 winMin = ImGui::GetWindowPos();

        bool textMode = !Config::signOutputToSpeech;
        ImVec2 tPos(winMin.x + 4.f, winMin.y + 4.f);
        ImVec2 tSize(iconSize - 8.f, iconSize - 8.f);
        ImGui::SetCursorScreenPos(tPos);
        ImGui::InvisibleButton("##sign_t", tSize);
        if (ImGui::IsItemClicked()) {
            Config::signOutputToSpeech = false;
            pendingAction_ = {HUDActionType::SetSignOutputText, -1};
        }
        ImU32 tCol = textMode ? IM_COL32(0, 200, 255, 255) : IM_COL32(120, 120, 120, 200);
        dl->AddText(ImVec2(tPos.x + 6.f, tPos.y + 2.f), tCol, "T");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Output to Text");

        ImVec2 sPos(winMin.x + iconSize + gap + 4.f, winMin.y + 4.f);
        ImVec2 sSize(iconSize - 8.f, iconSize - 8.f);
        ImGui::SetCursorScreenPos(sPos);
        ImGui::InvisibleButton("##sign_s", sSize);
        if (ImGui::IsItemClicked()) {
            Config::signOutputToSpeech = true;
            pendingAction_ = {HUDActionType::SetSignOutputSpeech, -1};
        }
        ImU32 sCol = !textMode ? IM_COL32(0, 200, 255, 255) : IM_COL32(120, 120, 120, 200);
        float cx = sPos.x + sSize.x * 0.5f, cy = sPos.y + sSize.y * 0.5f;
        dl->AddCircleFilled(ImVec2(cx - 3.f, cy), 3.f, sCol);
        dl->AddTriangleFilled(ImVec2(cx, cy - 4.f), ImVec2(cx, cy + 4.f), ImVec2(cx + 6.f, cy), sCol);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Output to Speech");
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

void HUDLayer::drawTTSOverlay(float vW, float vH, float yOff, float bottomReserved) {
    HUDSnapshot snap = gResultsStore.snapshot();
    if (!snap.ttsPlaying && snap.ttsSpokenText.empty()) return;

    const float overlayW = vW * 0.60f;
    const float overlayH = 52.f;
    const float overlayX = vW - overlayW - 8.f;
    const float overlayY = vH - overlayH - 8.f - bottomReserved + yOff;

    ImGui::SetNextWindowPos(ImVec2(overlayX, overlayY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(overlayW, overlayH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.95f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.12f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.f, 0.85f, 1.f, 1.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.f, 8.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoBringToFrontOnFocus;

    if (ImGui::Begin("##tts_overlay", nullptr, flags)) {
        if (snap.ttsPlaying) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 1.f, 0.4f, 1.f));
            ImGui::TextUnformatted("Speaking...");
            ImGui::PopStyleColor();
        }
        if (!snap.ttsSpokenText.empty()) {
            ImGui::SameLine(0.f, 12.f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.f, 0.85f, 0.9f, 0.8f));
            ImGui::TextUnformatted(snap.ttsSpokenText.c_str());
            ImGui::PopStyleColor();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

void HUDLayer::drawResolutionOverlay(float vW, float vH) {
    HUDSnapshot snap = gResultsStore.snapshot();
    char buf[80];
    std::snprintf(buf, sizeof(buf), "Capture: %d x %d  |  Detection: %d x %d",
                  snap.captureW, snap.captureH, snap.detectionW, snap.detectionH);
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    ImVec2 sz = ImGui::CalcTextSize(buf);
    float px = vW - sz.x - 12.f;
    float py = 8.f;
    if (px < 10.f) px = 10.f;
    dl->AddRectFilled({px - 4.f, py - 2.f}, {px + sz.x + 4.f, py + sz.y + 2.f},
                      IM_COL32(0, 0, 0, 120), 4.f);
    dl->AddText({px, py}, IM_COL32(180, 200, 220, 200), buf);
    (void)vH;
}

void HUDLayer::drawSTTTTSInputOverlay(float vW, float vH, float yOff, TranslationTaskManager* tm) {
    if (!tm) return;
    TranslationTask cur = tm->getCurrentTask();
    if (cur != TranslationTask::SPEECH_TO_TEXT && cur != TranslationTask::TEXT_TO_SPEECH)
        return;

    HUDSnapshot snap = gResultsStore.snapshot();
    const bool sapiOk = snap.sapiAvailable;

    // Semi-wide bar at bottom center, covering ~80% of width
    const float overlayW = vW * 0.80f;
    const float overlayH = (cur == TranslationTask::TEXT_TO_SPEECH) ? 90.f : 52.f;
    const float overlayX = (vW - overlayW) * 0.5f;
    // Raise slightly above absolute bottom (about 10% of height)
    const float overlayY = vH - overlayH - (vH * 0.10f) + yOff;

    // Position/size: only force on first frame after entering TTS, then let
    // ImGui preserve it (prevents focus resets and flicker).
    ImGuiCond posCond = ttsOverlayJustOpened_ ? ImGuiCond_Always : ImGuiCond_Once;
    ImGuiCond sizeCond = ttsOverlayJustOpened_ ? ImGuiCond_Always : ImGuiCond_Once;
    ImGui::SetNextWindowPos(ImVec2(overlayX, overlayY), posCond);
    ImGui::SetNextWindowSize(ImVec2(overlayW, overlayH), sizeCond);
    // Slight transparency so camera feed shows through
    ImGui::SetNextWindowBgAlpha(0.35f);
    // When first entering T2S, give overlay focus once so the text input can receive typing.
    if (cur == TranslationTask::TEXT_TO_SPEECH && ttsOverlayJustOpened_)
        ImGui::SetNextWindowFocus();
    ttsOverlayJustOpened_ = false;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.12f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.f, 0.85f, 1.f, 1.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.f, 8.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("##stt_tts_overlay", nullptr, flags)) {
        if (cur == TranslationTask::SPEECH_TO_TEXT) {
            const bool listening = snap.sttActive;
            const float micSize = 32.f;
            const float radius = 16.f;
            // Place mic button at bottom-left inside the horizontal bar
            ImGui::SetCursorPosY(ImGui::GetWindowHeight() - micSize - 8.f);
            ImGui::SetCursorPosX(8.f);
            ImGui::InvisibleButton("##stt_mic", ImVec2(micSize, micSize));
            bool micClicked = ImGui::IsItemClicked();
            bool micHover = ImGui::IsItemHovered();
            if (micClicked) {
                pendingAction_.type = listening ? HUDActionType::StopSTT : HUDActionType::StartSTT;
                pendingAction_.taskIdx = -1;
            }
            if (micHover)
                ImGui::SetTooltip("%s", listening ? "Stop Listening" : "Start Listening");

            ImVec2 rmin = ImGui::GetItemRectMin();
            ImVec2 rmax = ImGui::GetItemRectMax();
            ImVec2 center((rmin.x + rmax.x) * 0.5f, (rmin.y + rmax.y) * 0.5f);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImU32 fillCol = listening ? IM_COL32(0, 230, 102, 255) : IM_COL32(64, 64, 64, 230);
            ImU32 outCol = IM_COL32(140, 140, 140, 255);
            if (listening) {
                float pulse = 0.5f + 0.5f * std::sin(static_cast<float>(ImGui::GetTime() * 4.0));
                float r2 = radius + 2.f + pulse * 3.f;
                outCol = IM_COL32(0, 255, 128, 200);
                dl->AddCircle(center, r2, outCol, 0, 2.f);
            }
            dl->AddCircleFilled(center, radius, fillCol);
            dl->AddCircle(center, radius, outCol, 0, 1.5f);
            float cx = center.x, cy = center.y;
            float bw = 4.f, bh = 10.f;
            ImVec2 bodyMin(cx - bw * 0.5f, cy - bh * 0.5f);
            ImVec2 bodyMax(cx + bw * 0.5f, cy + bh * 0.5f);
            dl->AddRectFilled(bodyMin, bodyMax, IM_COL32(255, 255, 255, 255), 2.f);
            dl->AddLine(ImVec2(cx - 6.f, cy + bh * 0.5f), ImVec2(cx + 6.f, cy + bh * 0.5f), IM_COL32(255, 255, 255, 255), 1.5f);
            dl->AddLine(ImVec2(cx, cy + bh * 0.5f), ImVec2(cx, cy + 10.f), IM_COL32(255, 255, 255, 255), 1.5f);
            dl->AddRectFilled(ImVec2(cx - 4.f, cy + 10.f), ImVec2(cx + 4.f, cy + 12.f), IM_COL32(255, 255, 255, 255));

            ImGui::SameLine(0.f, 12.f);
            ImGui::SetCursorPosY(ImGui::GetWindowHeight() - micSize - 8.f + (micSize - ImGui::GetTextLineHeight()) * 0.5f);
            const char* txt = listening && snap.sttOriginal.empty()
                ? "Listening..." : snap.sttOriginal.empty() ? "Tap mic to speak" : snap.sttOriginal.c_str();
            ImGui::PushStyleColor(ImGuiCol_Text, (snap.sttOriginal.empty() && !listening) ? ImVec4(0.5f, 0.5f, 0.55f, 1.f) : ImVec4(1.f, 1.f, 1.f, 1.f));
            ImGui::TextUnformatted(txt);
            ImGui::PopStyleColor();

            if (!sapiOk || snap.sttStatus == "error") {
                ImGui::PushStyleColor(ImGuiCol_Text, kRed);
                ImGui::TextUnformatted("STT backend unavailable or error");
                ImGui::PopStyleColor();
            }
            // Space in STT overlay: Push-to-Talk (hold = listen, release = process)
            {
                static bool prevSpaceHeld = false;
                bool spaceHeld = ImGui::IsKeyDown(ImGuiKey_Space);

                // Press edge: start listening
                if (spaceHeld && !prevSpaceHeld && !snap.sttActive) {
                    pendingAction_.type = HUDActionType::StartSTT;
                }
                // Release edge: stop and process
                if (!spaceHeld && prevSpaceHeld && snap.sttActive) {
                    pendingAction_.type = HUDActionType::StopSTT;
                }

                prevSpaceHeld = spaceHeld;
            }
        } else {
            if (!snap.ttsSpokenText.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.f, 0.85f, 0.9f, 0.8f));
                ImGui::TextUnformatted(snap.ttsSpokenText.c_str());
                ImGui::PopStyleColor();
            }
            if (snap.ttsPlaying) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 1.f, 0.4f, 1.f));
                ImGui::TextUnformatted("Speaking...");
                ImGui::PopStyleColor();
            }

            // Static buffer â€” persists across frames, never cleared automatically
            static char ttsBuf[512] = {0};

            // Input fills most of the width, Speak sits at lower-right
            const float speakBtnW = 80.f;
            ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 36.f);
            ImGui::SetCursorPosX(8.f);
            ImGui::SetNextItemWidth(overlayW - speakBtnW - 24.f);

            // Always keep keyboard focus on the TTS input while in Text->Speech mode
            ImGui::SetKeyboardFocusHere(0);

            // Plain input â€” no per-frame checks, no trimming, no auto-clear
            bool enterPressed = ImGui::InputText(
                "##tts_input",
                ttsBuf,
                sizeof(ttsBuf),
                ImGuiInputTextFlags_EnterReturnsTrue
            );

            ImGui::SameLine(0.f, 8.f);
            ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 36.f);
            ImGui::SetCursorPosX(overlayW - speakBtnW - 8.f);

            // Speak button â€” always clickable, only reads ttsBuf at click time
            if (ImGui::Button("Speak", ImVec2(70.f, 0.f))) {
                if (ttsBuf[0] != '\0') {  // only check at click moment
                    HUDAction a;
                    a.type = HUDActionType::SpeakText;
                    a.text = std::string(ttsBuf);
                    a.taskIdx = -1;
                    pendingAction_ = a;
                    std::cout << "[HUD] TTS Speak button clicked, text='" << a.text << "'\n";
                }
            }

            // Enter key â€” only check at press moment
            if (enterPressed) {
                if (ttsBuf[0] != '\0') {
                    HUDAction a;
                    a.type = HUDActionType::SpeakText;
                    a.text = std::string(ttsBuf);
                    a.taskIdx = -1;
                    pendingAction_ = a;
                    std::cout << "[HUD] TTS Speak triggered, text='" << a.text << "'\n";
                }
            }

            // Space in TTS overlay: shortcut to Speak
            if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
                if (ttsBuf[0] != '\0') {
                    HUDAction a;
                    a.type = HUDActionType::SpeakText;
                    a.text = std::string(ttsBuf);
                    a.taskIdx = -1;
                    pendingAction_ = a;
                    std::cout << "[HUD] TTS Speak (Space) triggered, text='" << a.text << "'\n";
                }
            }

            if (!sapiOk) {
                ImGui::SetCursorPosY(6.f);
                ImGui::SetCursorPosX(8.f);
                ImGui::PushStyleColor(ImGuiCol_Text, kRed);
                ImGui::TextUnformatted("TTS backend unavailable");
                ImGui::PopStyleColor();
            }
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

void HUDLayer::drawSTTSubtitle(float vW, float vH, float yOff, bool showTranslation, float bottomReserved) {
    HUDSnapshot snap = gResultsStore.snapshot();

    const bool hasContent = !snap.sttOriginal.empty()   ||
                            !snap.sttTranslated.empty() ||
                            snap.sttActive;
    if (!hasContent) return;

    ImDrawList* dl  = ImGui::GetBackgroundDrawList();
    const float ph  = showTranslation ? 72.f : 44.f;
    const float px  = 10.f;
    const float py  = vH - ph - 10.f - bottomReserved - 8.f + yOff;
    const float pw  = vW - 20.f;

    dl->AddRectFilled({px, py}, {px + pw, py + ph}, IM_COL32(0, 0, 0, 170), 6.f);

    if (snap.sttActive && snap.sttOriginal.empty()) {
        dl->AddCircleFilled({px + 16.f, py + 22.f}, 6.f, IM_COL32(50, 220, 50, 255));
        dl->AddText({px + 28.f, py + 14.f}, IM_COL32(200, 255, 200, 255), "Listening...");
        return;
    }

    if (!snap.sttOriginal.empty())
        dl->AddText({px + 8.f, py + 8.f},
                    IM_COL32(180, 180, 180, 220), snap.sttOriginal.c_str());

    if (showTranslation && !snap.sttTranslated.empty())
        dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
                    {px + 8.f, py + 30.f},
                    IM_COL32(255, 255, 255, 255), snap.sttTranslated.c_str());
}

void HUDLayer::drawTTSIndicator() {
    HUDSnapshot snap = gResultsStore.snapshot();
    if (!snap.ttsPlaying) return;

    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    dl->AddRectFilled({8.f, 8.f}, {170.f, 32.f}, IM_COL32(0, 0, 0, 160), 4.f);
    dl->AddText({14.f, 12.f}, IM_COL32(100, 255, 100, 255), "Speaking...");
}

void HUDLayer::setRegistry(ModelRegistry* r) { registry_ = r; }

} // namespace SmartGlasses





