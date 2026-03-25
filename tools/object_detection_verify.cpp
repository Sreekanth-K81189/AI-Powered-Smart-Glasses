// Quick verify tool for object detection
#include <iostream>
#include "object/object_detector.h"
#include "utils/image_utils.h"
using namespace hud;
int main(int argc, char** argv) {
    std::string modelsPath = argc > 1 ? argv[1] : "models";
    std::string imagePath  = argc > 2 ? argv[2] : "";
    ObjectDetector det(modelsPath);
    cv::Mat frame = imagePath.empty() ?
        cv::Mat(480,640,CV_8UC3,cv::Scalar(128,128,128)) :
        LoadImage(imagePath);
    auto dets = det.Detect(frame);
    std::cout << "Detections: " << dets.size() << "\n";
    for (auto& d : dets)
        std::cout << "  " << d.className << " " << d.confidence << "\n";
    auto hint = det.GetMoveSafeHint(dets, frame.cols, frame.rows);
    std::cout << "MoveSafe: " << hint.message << "\n";
    return 0;
}
