#include "object/object_detector.h"
#include "object/object_store.h"
#include "utils/image_utils.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;
namespace hud {

static const std::vector<std::string> COCO80 = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

ObjectDetector::ObjectDetector(const std::string& modelsDir) {
    classNames_ = COCO80;
    // Project-standard YOLO location (kept model): models/yolo/yolov8x_fp16.onnx
    std::string modelPath = modelsDir + "/yolo/yolov8x_fp16.onnx";
    if (!fs::exists(modelPath)) {
        std::cerr << "[ObjectDetector] Model not found: " << modelPath << "\n";
        return;
    }
    try {
        // Force CPU execution provider; GPU provider DLLs are optional and
        // failure to load them should not crash the app.
        session_ = std::make_unique<OrtSession>(modelPath, false);
        std::cout << "[ObjectDetector] YOLOv8x_fp16 loaded\n";
    } catch (const std::exception& e) {
        std::cerr << "[ObjectDetector] " << e.what() << "\n";
    }
}

cv::Mat ObjectDetector::PreProcess(const cv::Mat& frame) {
    return ResizePad(frame, inputW_, inputH_);
}

std::vector<Detection> ObjectDetector::Detect(const cv::Mat& frame,
                                               float confThresh,
                                               float nmsThresh) {
    if (!session_ || frame.empty()) return {};
    cv::Mat padded = PreProcess(frame);

    std::vector<float> data;
    data.reserve(3 * inputH_ * inputW_);
    std::vector<cv::Mat> chans(3);
    cv::Mat f32;
    padded.convertTo(f32, CV_32FC3, 1.0f/255.0f);
    cv::split(f32, chans);
    for (int c=0;c<3;c++)
        for (int y=0;y<inputH_;y++)
            for (int x=0;x<inputW_;x++)
                data.push_back(chans[c].at<float>(y,x));

    std::array<int64_t,4> shape{1,3,(int64_t)inputH_,(int64_t)inputW_};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor  = Ort::Value::CreateTensor<float>(
        memInfo, data.data(), data.size(), shape.data(), shape.size());

    const char* inNames[]  = {"images"};
    const char* outNames[] = {"output0"};

    try {
        std::vector<Ort::Value> inputVec;
        inputVec.push_back(std::move(tensor));
        auto outputs = session_->Run({inNames,inNames+1},inputVec,{outNames,outNames+1});
        return PostProcess(outputs, frame.cols, frame.rows, confThresh, nmsThresh);
    } catch (const std::exception& e) {
        std::cerr << "[ObjectDetector] Inference error: " << e.what() << "\n";
        return {};
    }
}

std::vector<Detection> ObjectDetector::PostProcess(const std::vector<Ort::Value>& outputs,
                                                    int origW, int origH,
                                                    float confThresh, float nmsThresh) {
    // YOLOv8 output: [1, 84, 8400]
    auto* raw  = outputs[0].GetTensorData<float>();
    auto  dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int rows   = (int)dims[2]; // 8400
    int fields = (int)dims[1]; // 84 = 4 bbox + 80 classes

    float scaleX = (float)origW / inputW_;
    float scaleY = (float)origH / inputH_;

    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       classIds;

    for (int i = 0; i < rows; i++) {
        // Confidence = max class score
        float maxConf = 0; int maxCls = 0;
        for (int c = 4; c < fields; c++) {
            float v = raw[c * rows + i];
            if (v > maxConf) { maxConf = v; maxCls = c-4; }
        }
        if (maxConf < confThresh) continue;

        float cx = raw[0*rows+i] * scaleX;
        float cy = raw[1*rows+i] * scaleY;
        float bw = raw[2*rows+i] * scaleX;
        float bh = raw[3*rows+i] * scaleY;
        boxes.push_back({(int)(cx-bw/2),(int)(cy-bh/2),(int)bw,(int)bh});
        scores.push_back(maxConf);
        classIds.push_back(maxCls);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confThresh, nmsThresh, indices);

    std::vector<Detection> results;
    for (int idx : indices) {
        Detection d;
        d.bbox       = boxes[idx] & cv::Rect(0,0,origW,origH);
        d.confidence = scores[idx];
        d.classId    = classIds[idx];
        d.className  = (classIds[idx] < (int)classNames_.size()) ?
                        classNames_[classIds[idx]] : "unknown";
        results.push_back(d);
    }
    return results;
}

MoveSafeHint ObjectDetector::GetMoveSafeHint(const std::vector<Detection>& dets,
                                              int frameW, int frameH) {
    static const std::vector<std::string> obstacles = {
        "person","car","truck","bus","bicycle","motorcycle","chair","table",
        "potted plant","suitcase","dog","cat"
    };

    bool danger = false, caution = false;
    std::string label;

    int cx = frameW/2, w3 = frameW/3;

    for (auto& d : dets) {
        bool isObstacle = std::find(obstacles.begin(), obstacles.end(), d.className) != obstacles.end();
        if (!isObstacle) continue;

        int dcx = d.bbox.x + d.bbox.width/2;
        float area = (float)(d.bbox.width * d.bbox.height) / (frameW * frameH);

        if (area > 0.15f && dcx > w3 && dcx < 2*w3) {
            danger = true; label = d.className;
        } else if (area > 0.05f) {
            caution = true; label = d.className;
        }
    }

    if (danger)  return {"Obstacle ahead - slow down (" + label + ")", "danger"};
    if (caution) return {"Caution: " + label + " nearby", "caution"};
    return {"Path clear", "safe"};
}

void ObjectDetector::LoadCustomClass(const std::string& className,
                                      const std::vector<float>& proto) {
    classNames_.push_back(className);
    std::cout << "[ObjectDetector] Custom class added: " << className << "\n";
}

bool ObjectDetector::enrollObject(const std::vector<cv::Mat>& frames, const std::string& label) {
    if (frames.empty() || label.empty()) {
        spdlog::error("ObjectDetector::enrollObject: empty frames or label");
        return false;
    }
    try {
        hud::ObjectStore store("data/objects/object_store.json");
        std::vector<cv::Rect> boxes;
        for (size_t i = 0; i < frames.size(); ++i)
            boxes.push_back(cv::Rect(0, 0, frames[i].cols, frames[i].rows));
        store.AddClass(label, frames, boxes);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("ObjectDetector::enrollObject: {}", e.what());
        return false;
    }
}

} // namespace hud

