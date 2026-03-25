#pragma once

#include "object/ObjectEncoder.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace hud {

struct ObjectEntry {
    std::string              label;
    std::vector<ObjEmbedding> embeddings;
};

struct ObjectMatch {
    std::string label;
    float       similarity = 0.f;
    bool        matched    = false;
};

class ObjectRegistry {
public:
    explicit ObjectRegistry(const std::string& path);

    void AddObject(const std::string& label, const std::vector<ObjEmbedding>& embs);

    ObjectMatch FindBest(const ObjEmbedding& q, float threshold) const;

private:
    void Save() const;
    void Load();

    std::string          path_;
    std::vector<ObjectEntry> entries_;
};

} // namespace hud

