#include "hud/hud_layer.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace hud {

HudLayer::HudLayer() {}
HudLayer::~HudLayer() { Shutdown(); }

bool HudLayer::Init(int w, int h, const std::string& title) {
    winW_ = w; winH_ = h;
    if (!glfwInit()) { std::cerr << "[HUD] GLFW init failed\n"; return false; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(w, h, title.c_str(), nullptr, nullptr);
    if (!window_) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[HUD] GLAD init failed\n"; return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // Docking disabled - was causing Add Data panel to not receive clicks
    // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;  // Fresh layout every run - no saved docking to block clicks

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    glGenTextures(1, &videoTex_);
    std::cout << "[HUD] Initialized " << w << "x" << h << "\n";
    return true;
}

void HudLayer::UploadFrameToTexture(const cv::Mat& frame) {
    if (frame.empty()) return;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    glBindTexture(GL_TEXTURE_2D, videoTex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
}

void HudLayer::DrawVideoBackground() {
ImDrawList* dl = ImGui::GetBackgroundDrawList();
    if (videoTex_) {
        dl->AddImage(
            (ImTextureID)(intptr_t)videoTex_,
            ImVec2(0, 0),
            ImVec2((float)winW_, (float)winH_)
        );
    }
}

void HudLayer::DrawDetectionOverlays(const HudState& state) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    for (auto& d : state.detections) {
        ImU32 col = IM_COL32(0,255,0,200);
        if (d.className == "person") col = IM_COL32(255,165,0,200);
        dl->AddRect({(float)d.bbox.x,(float)d.bbox.y},
                    {(float)(d.bbox.x+d.bbox.width),(float)(d.bbox.y+d.bbox.height)},
                    col, 0, 0, 2.0f);
        std::string lbl = d.className + " " + std::to_string((int)(d.confidence*100)) + "%";
        dl->AddText({(float)d.bbox.x,(float)d.bbox.y-16}, col, lbl.c_str());
    }
    for (auto& f : state.faces) {
        dl->AddRect({(float)f.bbox.x,(float)f.bbox.y},
                    {(float)(f.bbox.x+f.bbox.width),(float)(f.bbox.y+f.bbox.height)},
                    IM_COL32(0,150,255,220), 0, 0, 2.0f);
        if (!f.label.empty()) {
            dl->AddText({(float)f.bbox.x,
                         (float)f.bbox.y - 18},
                        IM_COL32(0,200,255,255),
                        f.label.c_str());
        }
    }

    // Add Object manual selection rectangle (mouse drag on video)
    if (state.addObject.active) {
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 mp = io.MousePos;
        bool inside =
            mp.x >= 0.0f && mp.x <= (float)winW_ &&
            mp.y >= 0.0f && mp.y <= (float)winH_;

        static ImVec2 dragStart = {0,0};

        // Note: HudState is const here, so we only draw; selection state is
        // managed in a mutable copy via the non-const Render() / Add Data UI.
        if (inside && ImGui::IsMouseClicked(0)) {
            dragStart = mp;
        }
        if (ImGui::IsMouseDown(0)) {
            ImVec2 cur = mp;
            float x1 = std::min(dragStart.x, cur.x);
            float y1 = std::min(dragStart.y, cur.y);
            float x2 = std::max(dragStart.x, cur.x);
            float y2 = std::max(dragStart.y, cur.y);
            dl->AddRect({x1, y1}, {x2, y2}, IM_COL32(255,255,0,220), 0, 0, 2.0f);
        }
    }
    // MoveSafe overlay
    if (!state.moveSafeHint.empty()) {
        ImU32 hintCol = state.moveSafeHint.find("clear") != std::string::npos ?
                        IM_COL32(0,255,100,230) : IM_COL32(255,80,80,230);
        dl->AddRectFilled({10,(float)winH_-50},{(float)winW_-10,(float)winH_-10},
                          IM_COL32(0,0,0,160));
        dl->AddText({20,(float)winH_-42}, hintCol, state.moveSafeHint.c_str());
    }

    // Add Data hint - drawn to foreground list (no window, cannot block clicks)
    float hintY = (float)winH_ - 80.0f;
    dl->AddRectFilled({10.0f, hintY}, {410.0f, hintY + 70.0f}, IM_COL32(0,0,0,180));
    dl->AddText({20.0f, hintY + 8.0f}, IM_COL32(255,230,77,255),
                "Add Data: F2=Add Face | F3=Add Object | Esc=Back");
    if (state.addFace.active)
        dl->AddText({20.0f, hintY + 32.0f}, IM_COL32(77,255,128,255), ">>> ADD FACE ACTIVE <<<");
    else if (state.addObject.active)
        dl->AddText({20.0f, hintY + 32.0f}, IM_COL32(77,255,128,255), ">>> ADD OBJECT ACTIVE <<<");
}

// Fixed width for the right-side HUD panel (old interface layout)
static constexpr float kHudPanelWidth = 360.0f;

static const char* GetViewName(HudView v) {
    switch (v) {
        case HudView::DETECTION:   return "Detection View";
        case HudView::TRANSLATION: return "Translation Hub";
        case HudView::VOICE_ASSIST: return "Voice Assist";
        case HudView::ADD_DATA:    return "Add Data";
        case HudView::SETTINGS:    return "Settings";
        default: return "Menu";
    }
}

static int GetLayerNumber(HudView v) {
    switch (v) {
        case HudView::DETECTION:   return 1;
        case HudView::TRANSLATION: return 2;
        case HudView::VOICE_ASSIST: return 3;
        case HudView::ADD_DATA:    return 4;
        case HudView::SETTINGS:    return 5;
        default: return 1;
    }
}

void HudLayer::DrawHeaderBar(const HudState& state) {
    int layer = (state.currentView == HudView::TASK_MENU) ? 1 : GetLayerNumber(state.currentView);
    const char* viewName = (state.currentView == HudView::TASK_MENU) ? "Detection View" : GetViewName(state.currentView);

    // Header is aligned with the right-side HUD column, not full-width.
    float x = (float)winW_ - kHudPanelWidth;
    ImGui::SetNextWindowPos({x, 0.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({kHudPanelWidth, 40.0f}, ImGuiCond_Always);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.02f, 0.02f, 0.04f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.0f, 0.9f, 1.0f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);
    ImGui::Begin("##header", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);
    ImGui::SetCursorPosY(12.0f);
    char buf[128];
    snprintf(buf, sizeof(buf), " ? Layer %d/5 ? %s ? ? ", layer, viewName);
    ImGui::TextColored(ImVec4(0.0f, 0.95f, 1.0f, 1.0f), "%s", buf);
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
}

void HudLayer::DrawTaskMenu(HudState& state) {
    // Old layout: vertical task menu docked on the right edge.
    float x = (float)winW_ - kHudPanelWidth;
    ImGui::SetNextWindowPos({x, 40.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({kHudPanelWidth, (float)winH_ - 40.0f}, ImGuiCond_Always);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.06f, 0.08f, 0.95f));
    ImGui::Begin("##taskmenu", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);

    // Voice Feedback toggle - green when ON, clickable
    ImGui::PushStyleColor(ImGuiCol_Text, state.voiceFeedback ? ImVec4(0.2f, 1.0f, 0.4f, 1.0f) : ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
    if (state.voiceFeedback) {
        if (ImGui::Selectable("Voice Feedback: ON [click to Disable]", true, 0, ImVec2(-1, 0)))
            state.voiceFeedback = false;
    } else {
        if (ImGui::Selectable("Voice Feedback: OFF [click to Enable]", false, 0, ImVec2(-1, 0)))
            state.voiceFeedback = true;
    }
    ImGui::PopStyleColor();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextColored(ImVec4(0.0f, 0.9f, 1.0f, 1.0f), "SELECT TASK");
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.22f, 0.28f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.28f, 0.28f, 0.35f, 1.0f));

    float btnW = 320.0f;
    if (ImGui::Button("Object Detection", ImVec2(btnW, 40))) { state.currentView = HudView::DETECTION; state.detectionActive = true; }
    if (ImGui::Button("Face Detection", ImVec2(btnW, 40))) { state.currentView = HudView::DETECTION; state.faceActive = true; }
    if (ImGui::Button("Translation Hub", ImVec2(btnW, 40))) { state.currentView = HudView::TRANSLATION; }
    if (ImGui::Button("Voice Assist", ImVec2(btnW, 40))) { state.currentView = HudView::VOICE_ASSIST; state.voiceAssistActive = true; }
    if (ImGui::Button("Add Data", ImVec2(btnW, 40))) { state.currentView = HudView::ADD_DATA; state.addFace.active = false; state.addObject.active = false; }
    if (ImGui::Button("Settings", ImVec2(btnW, 40))) { state.currentView = HudView::SETTINGS; }

    ImGui::PopStyleColor(3);
    ImGui::End();
    ImGui::PopStyleColor(1);
}

void HudLayer::DrawMainPanel(const cv::Mat& frame, HudState& state) {
    if (state.currentView == HudView::TASK_MENU) {
        DrawTaskMenu(state);
        return;
    }

    // For non-menu views, reuse the same right-side HUD column.
    float x = (float)winW_ - kHudPanelWidth;
    ImGui::SetNextWindowPos({x, 40.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({kHudPanelWidth, (float)winH_ - 40.0f}, ImGuiCond_Always);
    ImGui::Begin("Main Control", nullptr, ImGuiWindowFlags_NoCollapse);

    // Back button to return to task menu
    if (ImGui::Button("<- Back to Menu", ImVec2(140, 24))) state.currentView = HudView::TASK_MENU;
    ImGui::Separator();

    ImGui::Text("SOURCE: %s", state.cameraSource.c_str());
    ImGui::Text("FPS:    %d", state.fps);
    ImGui::Separator();

    switch (state.currentView) {
        case HudView::DETECTION:
            ImGui::Checkbox("Object Detection",  &state.detectionActive);
            ImGui::Checkbox("Face Detection",    &state.faceActive);
            ImGui::Checkbox("OCR",               &state.ocrActive);
            break;
        case HudView::VOICE_ASSIST:
            ImGui::Checkbox("Voice Assist",      &state.voiceAssistActive);
            ImGui::Checkbox("Voice Feedback",    &state.voiceFeedback);
            break;
        case HudView::ADD_DATA:
            ImGui::Text("Add Data:");
            // Add Face / Add Object - highlighted when active
                    if (state.addFace.active) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.9f, 1.0f));
            }
            if (ImGui::Button("Add Face", ImVec2(140, 28))) {
                state.addFace.active = true;
                state.addObject.active = false;
            }
            if (state.addFace.active) ImGui::PopStyleColor();
            ImGui::SameLine();
            if (state.addObject.active) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.9f, 1.0f));
            }
            if (ImGui::Button("Add Object", ImVec2(140, 28))) {
                state.addObject.active = true;
                state.addFace.active = false;
            }
            if (state.addObject.active) ImGui::PopStyleColor();
            ImGui::SameLine();
            if (ImGui::Button("Back", ImVec2(60, 28))) {
                state.addFace.active = false;
                state.addObject.active = false;
            }
            if (state.addFace.active) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "ADD FACE (active)");
                ImGui::InputText("Name", state.addFace.name, sizeof(state.addFace.name));
                ImGui::Text("Captured: %d / 50", state.addFace.captured);
                ImGui::ProgressBar(state.addFace.captured / 50.0f, ImVec2(-1, 0));
                bool canCap = !frame.empty();
                if (!canCap) ImGui::BeginDisabled();
                if (ImGui::Button("Capture Face", ImVec2(-1, 0))) state.addFace.captureRequested = true;
                if (!canCap) ImGui::EndDisabled();
                ImGui::SameLine();
                bool canSave = state.addFace.name[0] != '\0' && state.addFace.captured >= 5;
                if (!canSave) ImGui::BeginDisabled();
                if (ImGui::Button("Save Identity")) state.addFace.saving = true;
                if (!canSave) ImGui::EndDisabled();
            } else if (state.addObject.active) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "ADD OBJECT (active)");
                ImGui::InputText("Class Name", state.addObject.name, sizeof(state.addObject.name));
                ImGui::Text("Samples: %d / 50", state.addObject.captured);
                bool canCap = !frame.empty();
                if (!canCap) ImGui::BeginDisabled();
                if (ImGui::Button("Capture Object", ImVec2(-1, 0))) state.addObject.captureRequested = true;
                if (!canCap) ImGui::EndDisabled();
                ImGui::SameLine();
                bool canSave = state.addObject.name[0] != '\0' && state.addObject.captured >= 5;
                if (!canSave) ImGui::BeginDisabled();
                if (ImGui::Button("Save Object Class")) state.addObject.saving = true;
                if (!canSave) ImGui::EndDisabled();
            }
            break;
        case HudView::TRANSLATION:
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "Translation Hub");
            ImGui::Text("Use STT and TTS for speech translation.");
            if (!state.taskOutput.empty())
                ImGui::TextWrapped("Output: %s", state.taskOutput.c_str());
            break;
        case HudView::SETTINGS: {
            ImGui::SliderFloat("Det Thresh",   &state.detConfThresh,  0.1f, 1.0f);
            ImGui::SliderFloat("Face Thresh",  &state.faceConfThresh, 0.1f, 1.0f);
            ImGui::SliderFloat("Match Thresh", &state.faceMatchThresh,0.3f, 1.0f);
            ImGui::SliderInt("Voice Interval", &state.voiceInterval,  1, 10);
            ImGui::Separator();
            static char esp32Buf[256] = {};
            static HudView prevView = HudView::TASK_MENU;
            if (prevView != HudView::SETTINGS) strncpy(esp32Buf, state.esp32Url.c_str(), 255);
            prevView = state.currentView;
            esp32Buf[255] = '\0';
            if (ImGui::InputText("ESP32 URL", esp32Buf, sizeof(esp32Buf)))
                state.esp32Url = esp32Buf;
            break;
        }
        default:
            break;
    }

    ImGui::Separator();
    if (!state.statusLine.empty()) {
        ImGui::TextWrapped("%s", state.statusLine.c_str());
    }
    ImGui::End();
}

void HudLayer::DrawSettingsPanel(HudState& state) {
    ImGui::SetNextWindowPos({510,10}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({240,200}, ImGuiCond_Once);
    ImGui::Begin("Settings");
    ImGui::SliderFloat("Det Thresh",   &state.detConfThresh,  0.1f, 1.0f);
    ImGui::SliderFloat("Face Thresh",  &state.faceConfThresh, 0.1f, 1.0f);
    ImGui::SliderFloat("Match Thresh", &state.faceMatchThresh,0.3f, 1.0f);
    ImGui::SliderInt("Voice Interval", &state.voiceInterval,  1, 10);
    ImGui::Separator();
    ImGui::InputText("ESP32 URL", &state.esp32Url[0], state.esp32Url.size()+64);
    ImGui::End();
}

void HudLayer::Render(const cv::Mat& frame, HudState& state) {
    glfwPollEvents();
    UploadFrameToTexture(frame);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    DrawVideoBackground();
    DrawDetectionOverlays(state);
    DrawHeaderBar(state);
    DrawMainPanel(frame, state);

    ImGui::Render();
    int fw, fh;
    glfwGetFramebufferSize(window_, &fw, &fh);
    glViewport(0, 0, fw, fh);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
}

bool HudLayer::ShouldClose() const { return glfwWindowShouldClose(window_); }

void HudLayer::Shutdown() {
    if (videoTex_) { glDeleteTextures(1, &videoTex_); videoTex_ = 0; }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (window_) { glfwDestroyWindow(window_); window_ = nullptr; }
    glfwTerminate();
}

} // namespace hud
