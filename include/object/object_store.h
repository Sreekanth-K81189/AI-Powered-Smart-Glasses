#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

namespace hud {

struct ObjectClass {
    std::string id;
    std::string name;
    std::vector<float>    protoEmbedding; // CLIP-style prototype
    std::vector<cv::Rect> sampleBoxes;   // annotation bounding boxes
    int    sampleCount = 0;
    float  confidenceThreshold = 0.6f;
};

class ObjectStore {
public:
    explicit ObjectStore(const std::string& dbPath);

    // Add new object class from annotated crops
    std::string AddClass(const std::string& name,
                         const std::vector<cv::Mat>& crops,
                         const std::vector<cv::Rect>& boxes);

    // Get prototype embedding for a class
    std::vector<float> GetPrototype(const std::string& classId);

    std::string FindMatch(const std::vector<float>& queryEmb, float threshold = 0.6f);

    void Save();
    void Load();

    std::vector<ObjectClass> GetAll() const;
    size_t Count() const { return classes_.size(); }

private:
    std::string dbPath_;
    std::unordered_map<std::string, ObjectClass> classes_;
    std::vector<float> ComputePrototype(const std::vector<cv::Mat>& crops);
    std::string GenerateUUID();
};

} // namespace hud
