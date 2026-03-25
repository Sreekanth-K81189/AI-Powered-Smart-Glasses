#pragma once
#include <string>
#include <vector>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>
#include <opencv2/core.hpp>
#include "Pipeline.hpp"

namespace SmartGlasses {

enum class TranslationTask {
  NONE,            // no translation task active (click same task again to turn off)
  SPEECH_TO_TEXT,
  TEXT_TO_SPEECH,
  OCR_TO_TEXT,
  OCR_TO_SPEECH,
  SIGN_TO_TEXT,
  SIGN_TO_SPEECH,
  TEXT_TO_SIGN,
  SPEECH_TO_SIGN,
  SCENE_TO_SPEECH,
  FACE_TO_SPEECH,
};

struct PipelineResult {
  std::string displayText;
  std::string transcribedText;
  std::vector<std::vector<float>> boxes;
  std::vector<cv::Rect> faces;
  float confidence = -1.f;
};

class TTSService;
class Pipeline;
class ModelRegistry;
class SignLanguageTranslator;

class TranslationTaskManager {
public:
  TranslationTaskManager(Pipeline& pipeline, TTSService& tts, ModelRegistry* registry = nullptr);
  ~TranslationTaskManager();
  void setRegistry(ModelRegistry* registry) { registry_ = registry; }
  void setTask(TranslationTask t);  // if t == current translation task, sets to NONE (turn off)
  TranslationTask getCurrentTask() const { return current_; }
  /** Call every frame when OCR task is active so the 500ms detection thread has a fresh frame. */
  void setLatestFrameForOCR(const cv::Mat& frame);
  void runCurrentTask(const cv::Mat& frame, PipelineResult& out);
  /** Run OCR once on the given frame; updates ResultsStore and speaks if task is OCR_TO_SPEECH. */
  void runOCRCapture(const cv::Mat& frame);
  /** Run OCR at a point (click): region crop or 200x100 crop; updates store and optional TTS. */
  void runOCRAtPoint(const cv::Mat& frame, int fx, int fy);
  /** Google Lens: 1st tap = capture image. Returns true if frame was stored. */
  void storeFrameForOCR(const cv::Mat& frame);
  /** True if we have a captured image waiting for 2nd tap (read aloud). */
  bool hasPendingOCRFrame() const;
  /** Google Lens: 2nd tap = run OCR on stored frame + T2S. Call after hasPendingOCRFrame(). */
  void runOCROnPendingAndSpeak();
  // Run object (scene) and face detection together, controlled by flags.
  void runDetectionTasks(const cv::Mat& frame,
                         bool runScene,
                         bool runFace,
                         PipelineResult& out);
  std::string getTaskLabel(TranslationTask t) const;
  std::vector<TranslationTask> getAllTasks() const;

private:
  TranslationTask current_ = TranslationTask::NONE;
  Pipeline* pipeline_ = nullptr;
  TTSService* tts_ = nullptr;
  ModelRegistry* registry_ = nullptr;

  std::unique_ptr<SignLanguageTranslator> signLT_;

  std::mutex ocrFrameMutex_;
  cv::Mat latestOcrFrame_;
  cv::Mat pendingOcrFrame_;       // Google Lens: stored on 1st tap, OCR on 2nd
  std::atomic<bool> hasPendingOcrFrame_{ false };
  std::atomic<bool> ocrDetectStop_{ true };
  std::thread ocrDetectThread_;
  void ocrDetectThreadFunc_();

  void sceneToSpeech_(const cv::Mat& frame, PipelineResult& out);
  void faceToSpeech_(const cv::Mat& frame, PipelineResult& out);
  void textToSign_(const std::string& text, PipelineResult& out);
};

} // namespace SmartGlasses
