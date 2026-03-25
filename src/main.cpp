#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "CameraManager.hpp"
#include "ModelRegistry.hpp"
#include "Pipeline.hpp"
#include "TTSService.hpp"
#include "HUDLayer.hpp"
#include "Config.hpp"
#include "TranslationTaskManager.hpp"
#include "PythonBridge.hpp"
#include "STTEngine.h"
#include <spdlog/spdlog.h>

#include <filesystem>
#include <cstdlib>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <chrono>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace SmartGlasses {
GLFWwindow* gHudWindow = nullptr;
}

static void glfwErrorCallback(int code, const char* msg) {
    spdlog::warn("GLFW {}: {}", code, msg);
}

int main(int argc, char** argv) {
    (void)argc;
    std::string exeDir;
#ifdef _WIN32
    char path[MAX_PATH];
    if (GetModuleFileNameA(nullptr, path, MAX_PATH)) {
        exeDir = path;
        size_t last = exeDir.find_last_of("\\/");
        if (last != std::string::npos) exeDir = exeDir.substr(0, last);
    }
#else
    if (argv[0]) {
        exeDir = argv[0];
        size_t last = exeDir.find_last_of("/");
        if (last != std::string::npos) exeDir = exeDir.substr(0, last);
    }
#endif
    if (exeDir.empty()) exeDir = ".";
    namespace fs = std::filesystem;
    std::string modelsDir = exeDir + "/models";

    // Prefer models folder next to the executable. Only fall back if it truly
    // does not exist; this avoids accidentally pointing at stale deps paths
    // via VISION_MODELS and then not seeing the ONNX / cascade files that are
    // copied beside the built .exe.
    if (!fs::exists(modelsDir)) {
        if (const char* env = std::getenv("VISION_MODELS");
            env != nullptr && fs::exists(env)) {
            modelsDir = env;
        } else {
            modelsDir = exeDir + "/../../models";
        }
    }
    if (!fs::exists(modelsDir))
        spdlog::error("models dir not found: {}", modelsDir);
    SmartGlasses::Config::modelsDir = modelsDir;

    SmartGlasses::PythonBridge::setBaseDir(exeDir);

    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Smart Glasses HUD", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    SmartGlasses::gHudWindow = window;

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int w, int h) {
        glViewport(0, 0, w, h);
    });

    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int action, int) {
        if (action != GLFW_PRESS) return;
        if (key == GLFW_KEY_T)
            SmartGlasses::Config::TTS_ENABLED = !SmartGlasses::Config::TTS_ENABLED;
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(w, GLFW_TRUE);
    });

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding   = 0.0f;
    s.FrameRounding    = 4.0f;
    s.WindowBorderSize = 1.0f;
    s.FramePadding     = {8, 5};
    s.ItemSpacing      = {10, 6};
    s.Colors[ImGuiCol_WindowBg]      = {0.07f, 0.07f, 0.09f, 1.0f};
    s.Colors[ImGuiCol_TitleBgActive] = {0.10f, 0.35f, 0.60f, 1.0f};
    s.Colors[ImGuiCol_Button]        = {0.13f, 0.38f, 0.62f, 1.0f};
    s.Colors[ImGuiCol_ButtonHovered] = {0.18f, 0.52f, 0.80f, 1.0f};
    s.Colors[ImGuiCol_Header]        = {0.13f, 0.38f, 0.62f, 0.8f};
    s.Colors[ImGuiCol_HeaderHovered] = {0.18f, 0.52f, 0.80f, 1.0f};
    s.Colors[ImGuiCol_FrameBg]       = {0.11f, 0.11f, 0.14f, 1.0f};
    s.Colors[ImGuiCol_Separator]     = {0.25f, 0.25f, 0.30f, 1.0f};
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    spdlog::info("Loading models...");
    SmartGlasses::ModelRegistry registry;
    bool modelsOk = registry.initialize(SmartGlasses::Config::modelsDir,
        [](int step, const std::string& name) {
            spdlog::info("  [{}] {}", step, name);
        });
    if (!modelsOk)
        spdlog::warn("Some models failed to load.");

    SmartGlasses::CameraManager camera;
    if (!camera.startFeed()) {
        spdlog::warn("Camera not ready — continuing without live feed (window will show No Signal)");
    }

    SmartGlasses::TTSService tts;
    SmartGlasses::Pipeline pipeline(camera, registry, tts);
    SmartGlasses::TranslationTaskManager taskMgr(pipeline, tts, &registry);
    SmartGlasses::HUDLayer hud;
    hud.setRegistry(&registry);

    std::mutex frameMutex;
    cv::Mat displayFrame;
    std::atomic<bool> running{ true };

    std::mutex detMutex;
    std::condition_variable detCV;
    cv::Mat detInputFrame;
    bool detFrameReady = false;
    SmartGlasses::PipelineResult detLatestResult;
    bool detResultReady = false;
    bool detDetectionActive = false;
    bool detFaceDetectionActive = false;

    std::thread captureThread([&]() {
        while (running.load()) {
            cv::Mat frame;
            camera.readFrame(frame);
            if (!frame.empty()) {
                std::lock_guard<std::mutex> lock(frameMutex);
                displayFrame = frame.clone();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    });

    std::thread detectionThread([&]() {
        while (running.load()) {
            cv::Mat frame;
            bool runScene = false;
            bool runFace = false;
            {
                std::unique_lock<std::mutex> lock(detMutex);
                detCV.wait_for(lock, std::chrono::milliseconds(100), [&]() {
                    return detFrameReady || !running.load();
                });
                if (!running.load()) break;
                if (!detFrameReady) continue;
                frame = detInputFrame.clone();
                runScene = detDetectionActive;
                runFace = detFaceDetectionActive;
                detFrameReady = false;
            }
            if (!frame.empty() && (runScene || runFace)) {
                SmartGlasses::PipelineResult out;
                taskMgr.runDetectionTasks(frame, runScene, runFace, out);
                std::lock_guard<std::mutex> lock(detMutex);
                detLatestResult = std::move(out);
                detResultReady = true;
            }
        }
    });

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int W, H;
        glfwGetFramebufferSize(window, &W, &H);

        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!displayFrame.empty())
                frame = displayFrame.clone();
        }

        std::string statusLine = camera.getStatus();
        std::string hintLine = "Ready";
        float confidencePercent = 0.f;
        std::vector<std::vector<float>> detectionBoxes;
        std::vector<cv::Rect> faceRects;

        if (!frame.empty()) {
            {
                std::lock_guard<std::mutex> lock(detMutex);
                detDetectionActive = hud.isDetectionActive();
                detFaceDetectionActive = hud.isFaceDetectionActive();
                detInputFrame = frame.clone();
                detFrameReady = true;
            }
            detCV.notify_one();

            SmartGlasses::TranslationTask currentTask = taskMgr.getCurrentTask();
            const bool useDetectionResult = (currentTask == SmartGlasses::TranslationTask::SCENE_TO_SPEECH ||
                                            currentTask == SmartGlasses::TranslationTask::FACE_TO_SPEECH);

            {
                std::lock_guard<std::mutex> lock(detMutex);
                if (detResultReady) {
                    detectionBoxes = detLatestResult.boxes;
                    faceRects = detLatestResult.faces;
                    if (useDetectionResult) {
                        if (!detLatestResult.displayText.empty())
                            hintLine = detLatestResult.displayText;
                        if (detLatestResult.confidence >= 0.f)
                            confidencePercent = detLatestResult.confidence * 100.f;
                    }
                    detResultReady = false;
                }
            }

            // OCR is on-demand only (two-tap: capture then read) — no per-frame feed.

            SmartGlasses::PipelineResult taskResult;
            if (!useDetectionResult) {
                taskMgr.runCurrentTask(frame, taskResult);
                if (!taskResult.displayText.empty())
                    hintLine = taskResult.displayText;
                if (taskResult.confidence >= 0.f)
                    confidencePercent = taskResult.confidence * 100.f;
            }
        }

        float fps = ImGui::GetIO().Framerate;
        hud.draw((float)W, (float)H, frame, statusLine, hintLine, confidencePercent,
                 detectionBoxes, faceRects, fps, &taskMgr);

        for (;;) {
            SmartGlasses::HUDAction action = hud.pollAction();
            if (action.type == SmartGlasses::HUDActionType::None) break;

            switch (action.type) {
                case SmartGlasses::HUDActionType::SetTask: {
                    // HUD sends the TranslationTask enum value, not an index into getAllTasks().
                    if (action.taskIdx >= 0 && action.taskIdx <= (int)SmartGlasses::TranslationTask::FACE_TO_SPEECH)
                        taskMgr.setTask(static_cast<SmartGlasses::TranslationTask>(action.taskIdx));
                    break;
                }
                case SmartGlasses::HUDActionType::SpeakText:
                    if (!action.text.empty())
                        tts.speak(action.text, true);  // forceUserAction = true so Speak always outputs
                    break;
                case SmartGlasses::HUDActionType::StartSTT:
                    gSTTEngine.startListening(true);  // push-to-talk: hold Space to record
                    break;
                case SmartGlasses::HUDActionType::StopSTT:
                    gSTTEngine.stopListening();
                    break;
                case SmartGlasses::HUDActionType::CaptureOCR:
                    if (!frame.empty()) {
                        if (taskMgr.hasPendingOCRFrame())
                            taskMgr.runOCROnPendingAndSpeak();
                        else
                            taskMgr.storeFrameForOCR(frame);
                    }
                    break;
                case SmartGlasses::HUDActionType::SetVoiceFeedback:
                    hud.setVoiceFeedback(!hud.isVoiceFeedbackEnabled());
                    break;
                case SmartGlasses::HUDActionType::AddFace:
                    pipeline.captureEnrollmentFrames(
                        action.taskIdx, action.text, SmartGlasses::EnrollmentType::FACE,
                        [](float){},
                        [](bool ok){ spdlog::info("Face enroll {}", ok ? "done" : "failed"); });
                    break;
                case SmartGlasses::HUDActionType::AddObject:
                    pipeline.captureEnrollmentFrames(
                        action.taskIdx, action.text, SmartGlasses::EnrollmentType::OBJECT,
                        [](float){},
                        [](bool ok){ spdlog::info("Object enroll {}", ok ? "done" : "failed"); });
                    break;
                default:
                    break;
            }
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    running.store(false);
    detCV.notify_all();
    if (captureThread.joinable()) captureThread.join();
    if (detectionThread.joinable()) detectionThread.join();

    SmartGlasses::gHudWindow = nullptr;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
