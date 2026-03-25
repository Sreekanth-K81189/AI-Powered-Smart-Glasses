#include "ResultsStore.h"

#include <chrono>

static int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void ResultsStore::setBoxes(std::vector<DetectionBox> boxes) {
    std::lock_guard<std::mutex> lock(mtx_);
    boxes_ = std::move(boxes);
}

void ResultsStore::setFaces(std::vector<FaceResult> faces) {
    std::lock_guard<std::mutex> lock(mtx_);
    faces_ = std::move(faces);
}

void ResultsStore::setSTTOriginal(const std::string& text, const std::string& lang) {
    std::lock_guard<std::mutex> lock(mtx_);
    sttOriginal_ = text;
    sttTimestamp_ = nowMs();
    (void)lang;
}

void ResultsStore::setSTTTranslated(const std::string& text) {
    std::lock_guard<std::mutex> lock(mtx_);
    sttTranslated_ = text;
}

void ResultsStore::setSTTActive(bool active) {
    std::lock_guard<std::mutex> lock(mtx_);
    sttActive_ = active;
}

void ResultsStore::setOCROriginal(const std::string& text) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrOriginal_ = text;
    ocrTimestamp_ = nowMs();
}

void ResultsStore::setOCRTranslated(const std::string& text, bool visible) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrTranslated_ = text;
    ocrVisible_ = visible;
    ocrTimestamp_ = nowMs();
}

void ResultsStore::setTTSPlaying(bool playing) {
    std::lock_guard<std::mutex> lock(mtx_);
    ttsPlaying_ = playing;
    if (!playing) ttsSpokenText_.clear();
}

void ResultsStore::setTTSSpokenText(const std::string& text) {
    std::lock_guard<std::mutex> lock(mtx_);
    ttsSpokenText_ = text;
}

void ResultsStore::setFeedInfo(int captureW, int captureH, int detW, int detH) {
    std::lock_guard<std::mutex> lock(mtx_);
    captureW_ = captureW;
    captureH_ = captureH;
    detectionW_ = detW;
    detectionH_ = detH;
}

void ResultsStore::setSAPIAvailable(bool available) {
    std::lock_guard<std::mutex> lock(mtx_);
    sapiAvailable_ = available;
}

void ResultsStore::setSTTStatus(const std::string& status) {
    std::lock_guard<std::mutex> lock(mtx_);
    sttStatus_ = status;
}

void ResultsStore::setSTTPartial(const std::string& partial) {
    std::lock_guard<std::mutex> lock(mtx_);
    sttPartial_ = partial;
}

void ResultsStore::setOCRStatus(const std::string& status) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrStatus_ = status;
}

void ResultsStore::setOCRLiveOn(bool on) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrLiveOn_ = on;
}
bool ResultsStore::getOCRLiveOn() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return ocrLiveOn_;
}
void ResultsStore::setOCRCropMode(bool on) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrCropMode_ = on;
}
bool ResultsStore::getOCRCropMode() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return ocrCropMode_;
}
void ResultsStore::setOCRCaptureRequested(bool req) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrCaptureRequested_ = req;
}
bool ResultsStore::getOCRCaptureRequested() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return ocrCaptureRequested_;
}
void ResultsStore::setOCRImageCaptured(bool captured) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrImageCaptured_ = captured;
}
bool ResultsStore::getOCRImageCaptured() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return ocrImageCaptured_;
}
void ResultsStore::setOCRUseCrop(bool use) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrUseCrop_ = use;
}
void ResultsStore::setOCRCropRegion(cv::Rect r) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrCropRegion_ = r;
}
cv::Rect ResultsStore::getOCRCropRegion() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return ocrCropRegion_;
}
void ResultsStore::setLastOcrScanTime(std::chrono::steady_clock::time_point t) {
    std::lock_guard<std::mutex> lock(mtx_);
    lastOcrScanTime_ = t;
}
std::chrono::steady_clock::time_point ResultsStore::getLastOcrScanTime() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return lastOcrScanTime_;
}
void ResultsStore::setLastOcrSpeakTime(std::chrono::steady_clock::time_point t) {
    std::lock_guard<std::mutex> lock(mtx_);
    lastOcrSpeakTime_ = t;
}
std::chrono::steady_clock::time_point ResultsStore::getLastOcrSpeakTime() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return lastOcrSpeakTime_;
}
void ResultsStore::setLastSpokenOcr(const std::string& s) {
    std::lock_guard<std::mutex> lock(mtx_);
    lastSpokenOcr_ = s;
}
std::string ResultsStore::getLastSpokenOcr() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return lastSpokenOcr_;
}
void ResultsStore::setOCRBoxesRotated(const std::vector<cv::RotatedRect>& boxes) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrBoxesRotated_ = boxes;
}
void ResultsStore::setOCRConfidence(float c) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrConfidence_ = c;
}
float ResultsStore::getOCRConfidence() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return ocrConfidence_;
}

void ResultsStore::setTTSStatus(const std::string& status) {
    std::lock_guard<std::mutex> lock(mtx_);
    ttsStatus_ = status;
}

void ResultsStore::setOCRRegions(const std::vector<OCRRegion>& regions) {
    std::lock_guard<std::mutex> lock(mtx_);
    ocrRegions_ = regions;
}

void ResultsStore::setSelectedOCRRegion(int index, const std::string& text) {
    std::lock_guard<std::mutex> lock(mtx_);
    selectedOCRRegion_ = index;
    selectedOCRText_   = text;
    ocrOverlayTimer_   = 8.0f;
}

void ResultsStore::tickOCRTimers(float dt) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (ocrOverlayTimer_ > 0.f) {
        ocrOverlayTimer_ -= dt;
        if (ocrOverlayTimer_ <= 0.f) {
            ocrOverlayTimer_   = 0.f;
            selectedOCRRegion_ = -1;
            selectedOCRText_.clear();
        }
    }
}

void ResultsStore::setSignWord(const std::string& word) {
    std::lock_guard<std::mutex> lock(mtx_);
    signWord_ = word;
}

std::string ResultsStore::getSignWord() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return signWord_;
}

void ResultsStore::setSignConfidence(float confidence) {
    std::lock_guard<std::mutex> lock(mtx_);
    signConfidence_ = confidence;
}

float ResultsStore::getSignConfidence() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return signConfidence_;
}

HUDSnapshot ResultsStore::snapshot() const {
    std::lock_guard<std::mutex> lock(mtx_);

    HUDSnapshot s;
    s.boxes = boxes_;
    s.faces = faces_;

    s.sttOriginal = sttOriginal_;
    s.sttTranslated = sttTranslated_;
    s.sttPartial = sttPartial_;
    s.sttTimestamp = sttTimestamp_;
    s.sttActive = sttActive_;

    s.ocrOriginal = ocrOriginal_;
    s.ocrTranslated = ocrTranslated_;
    s.ocrVisible = ocrVisible_;
    s.ocrTimestamp = ocrTimestamp_;
    s.ocrConfidence = ocrConfidence_;
    s.ocrBoxesRotated = ocrBoxesRotated_;
    s.ocrRegions = ocrRegions_;
    s.ocrLiveOn = ocrLiveOn_;
    s.ocrCropMode = ocrCropMode_;
    s.ocrCaptureRequested = ocrCaptureRequested_;
    s.ocrImageCaptured = ocrImageCaptured_;
    s.ocrUseCrop = ocrUseCrop_;
    s.ocrCropRegion = ocrCropRegion_;
    s.selectedOCRRegion = selectedOCRRegion_;
    s.ocrOverlayTimer = ocrOverlayTimer_;
    s.selectedOCRText = selectedOCRText_;

    s.ttsPlaying = ttsPlaying_;
    s.ttsSpokenText = ttsSpokenText_;

    s.captureW = captureW_;
    s.captureH = captureH_;
    s.detectionW = detectionW_;
    s.detectionH = detectionH_;
    s.sapiAvailable = sapiAvailable_;
    s.sttStatus = sttStatus_;
    s.ocrStatus = ocrStatus_;
    s.ttsStatus = ttsStatus_;

    return s;
}

void ResultsStore::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    boxes_.clear();
    faces_.clear();

    sttOriginal_.clear();
    sttTranslated_.clear();
    sttPartial_.clear();
    ocrOriginal_.clear();
    ocrTranslated_.clear();

    ocrVisible_ = false;
    ocrConfidence_ = 0.f;
    ocrBoxesRotated_.clear();
    ocrLiveOn_ = false;
    ocrCropMode_ = false;
    ocrCaptureRequested_ = false;
    ocrUseCrop_ = false;
    lastSpokenOcr_.clear();
    ttsPlaying_ = false;
    ttsSpokenText_.clear();
    sttStatus_ = "disabled";
    ocrStatus_ = "disabled";
    ttsStatus_ = "disabled";
    ocrRegions_.clear();
    selectedOCRRegion_ = -1;
    ocrOverlayTimer_   = 0.f;
    selectedOCRText_.clear();
    signWord_.clear();
    signConfidence_ = 0.f;
}

ResultsStore gResultsStore;

