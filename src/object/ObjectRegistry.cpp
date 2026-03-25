#include "object/ObjectRegistry.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace hud {

ObjectRegistry::ObjectRegistry(const std::string& path)
    : path_(path) {
    Load();
}

void ObjectRegistry::AddObject(const std::string& label, const std::vector<ObjEmbedding>& embs) {
    if (embs.empty()) return;
    ObjectEntry e;
    e.label = label;
    e.embeddings = embs;
    entries_.push_back(std::move(e));
    Save();
}

ObjectMatch ObjectRegistry::FindBest(const ObjEmbedding& q, float threshold) const {
    ObjectMatch best;
    if (q.empty()) return best;
    for (const auto& e : entries_) {
        for (const auto& emb : e.embeddings) {
            float sim = ObjectEncoder::Cosine(q, emb);
            if (sim > best.similarity) {
                best.similarity = sim;
                best.label      = e.label;
            }
        }
    }
    best.matched = (!best.label.empty() && best.similarity >= threshold);
    return best;
}

void ObjectRegistry::Save() const {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& e : entries_) {
        nlohmann::json je;
        je["label"] = e.label;
        je["embeddings"] = e.embeddings;
        j.push_back(je);
    }
    try {
        std::filesystem::path p(path_);
        auto dir = p.parent_path();
        if (!dir.empty())
            std::filesystem::create_directories(dir);
    } catch (const std::exception& ex) {
        std::cerr << "[ObjectRegistry] mkdir error: " << ex.what() << "\n";
    }
    std::ofstream f(path_);
    if (f.is_open()) {
        f << j.dump(2);
    }
}

void ObjectRegistry::Load() {
    entries_.clear();
    std::ifstream f(path_);
    if (!f.is_open()) return;
    try {
        nlohmann::json j;
        f >> j;
        for (auto& je : j) {
            ObjectEntry e;
            e.label = je.value("label", std::string());
            if (je.contains("embeddings")) {
                e.embeddings = je["embeddings"].get<std::vector<ObjEmbedding>>();
            }
            if (!e.label.empty() && !e.embeddings.empty())
                entries_.push_back(std::move(e));
        }
        std::cout << "[ObjectRegistry] Loaded " << entries_.size() << " entries\n";
    } catch (const std::exception& ex) {
        std::cerr << "[ObjectRegistry] Load error: " << ex.what() << "\n";
    }
}

} // namespace hud

