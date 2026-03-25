#include "object/object_store.h"
#include "utils/image_utils.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <random>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace hud {

ObjectStore::ObjectStore(const std::string& dbPath)
    : dbPath_(dbPath) {
    Load();
}

std::string ObjectStore::GenerateUUID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 15);
    auto hex = [](int v) { return "0123456789abcdef"[v & 0xF]; };

    std::string s(36, ' ');
    for (int i : {0,1,2,3,4,5,6,7,9,10,11,12,14,15,16,17,
                  19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35}) {
        s[i] = hex(dist(gen));
    }
    s[8]  = s[13] = s[18] = s[23] = '-';
    s[14] = '4';
    s[19] = "89ab"[dist(gen) % 4];
    return s;
}

std::vector<float> ObjectStore::ComputePrototype(const std::vector<cv::Mat>& crops) {
    if (crops.empty()) return {};

    std::vector<float> acc;
    int count = 0;

    for (const auto& img : crops) {
        if (img.empty()) continue;
        auto v = MatToVector(img);
        if (v.empty()) continue;
        if (acc.empty()) {
            acc.assign(v.size(), 0.0f);
        }
        if (v.size() != acc.size()) continue;
        for (size_t i = 0; i < v.size(); ++i) {
            acc[i] += v[i];
        }
        ++count;
    }

    if (count == 0) return {};
    for (auto& x : acc) x /= static_cast<float>(count);
    return acc;
}

std::string ObjectStore::AddClass(const std::string& name,
                                  const std::vector<cv::Mat>& crops,
                                  const std::vector<cv::Rect>& boxes) {
    ObjectClass cls;
    cls.id           = GenerateUUID();
    cls.name         = name;
    cls.protoEmbedding = ComputePrototype(crops);
    cls.sampleBoxes  = boxes;
    cls.sampleCount  = static_cast<int>(crops.size());

    classes_[cls.id] = cls;
    Save();
    return cls.id;
}

std::vector<float> ObjectStore::GetPrototype(const std::string& classId) {
    auto it = classes_.find(classId);
    if (it == classes_.end()) return {};
    return it->second.protoEmbedding;
}

std::string ObjectStore::FindMatch(const std::vector<float>& queryEmb, float threshold) {
    if (queryEmb.empty()) return {};

    auto cosine = [](const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        if (na <= 0.0f || nb <= 0.0f) return 0.0f;
        return dot / std::sqrt(na * nb);
    };

    std::string bestId;
    float bestSim = 0.0f;

    for (auto& [id, cls] : classes_) {
        float sim = cosine(queryEmb, cls.protoEmbedding);
        if (sim > bestSim) {
            bestSim = sim;
            bestId  = id;
        }
    }

    if (bestSim >= threshold) return bestId;
    return {};
}

void ObjectStore::Save() {
    nlohmann::json j = nlohmann::json::array();
    for (auto& [id, cls] : classes_) {
        nlohmann::json entry;
        entry["id"]        = cls.id;
        entry["name"]      = cls.name;
        entry["proto"]     = cls.protoEmbedding;
        entry["threshold"] = cls.confidenceThreshold;
        entry["samples"]   = cls.sampleCount;

        nlohmann::json boxes = nlohmann::json::array();
        for (const auto& r : cls.sampleBoxes) {
            boxes.push_back({
                {"x", r.x},
                {"y", r.y},
                {"w", r.width},
                {"h", r.height}
            });
        }
        entry["boxes"] = boxes;
        j.push_back(entry);
    }

    std::ofstream f(dbPath_);
    if (f.is_open()) {
        f << j.dump(2);
    }
}

void ObjectStore::Load() {
    classes_.clear();
    std::ifstream f(dbPath_);
    if (!f.is_open()) return;

    try {
        nlohmann::json j;
        f >> j;
        for (auto& entry : j) {
            ObjectClass cls;
            cls.id   = entry.value("id", "");
            cls.name = entry.value("name", "");
            cls.protoEmbedding = entry.value("proto", std::vector<float>{});
            cls.confidenceThreshold = entry.value("threshold", 0.6f);
            cls.sampleCount = entry.value("samples", 0);

            cls.sampleBoxes.clear();
            if (entry.contains("boxes")) {
                for (auto& b : entry["boxes"]) {
                    int x = b.value("x", 0);
                    int y = b.value("y", 0);
                    int w = b.value("w", 0);
                    int h = b.value("h", 0);
                    cls.sampleBoxes.emplace_back(x, y, w, h);
                }
            }

            if (!cls.id.empty()) {
                classes_[cls.id] = std::move(cls);
            }
        }
    } catch (...) {
        // ignore malformed JSON for now
    }
}

std::vector<ObjectClass> ObjectStore::GetAll() const {
    std::vector<ObjectClass> out;
    out.reserve(classes_.size());
    for (const auto& kv : classes_) {
        out.push_back(kv.second);
    }
    return out;
}

} // namespace hud

