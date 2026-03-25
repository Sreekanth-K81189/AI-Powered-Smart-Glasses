#include "face/identity_store.h"
#include "face/face_encoder.h"
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <random>
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace hud {

IdentityStore::IdentityStore(const std::string& dbPath) : dbPath_(dbPath) {
    Load();
}
IdentityStore::~IdentityStore() { Save(); }

std::string IdentityStore::GenerateUUID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> d(0, 15);
    auto t = [&](int n){ return "0123456789abcdef"[d(gen)]; };
    std::string s(36,' ');
    for (int i:{ 0,1,2,3,4,5,6,7,9,10,11,12,14,15,16,17,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35})
        s[i]=t(0);
    s[8]=s[13]=s[18]=s[23]='-'; s[14]='4';
    s[19]="89ab"[d(gen)%4];
    return s;
}

std::string IdentityStore::AddIdentity(const std::string& name,
                                        const Embedding& embedding,
                                        const std::vector<std::string>& tags,
                                        float threshold) {
    Identity id;
    id.id        = GenerateUUID();
    id.name      = name;
    id.embedding = embedding;
    id.tags      = tags;
    id.confidenceThreshold = threshold;

    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
    id.firstSeen = id.lastSeen = ss.str();

    identities_[id.id] = id;
    Save();
    std::cout << "[IdentityStore] Added: " << name << " (" << id.id << ")\n";
    return id.id;
}

MatchResult IdentityStore::FindMatch(const Embedding& query, float threshold) {
    MatchResult best;
    for (auto& [uid, ident] : identities_) {
        float sim = FaceEncoder::CosineSimilarity(query, ident.embedding);
        if (sim > best.similarity) {
            best.similarity = sim;
            best.identity   = ident;
        }
    }
    float thresh = threshold > 0 ? threshold :
                   (best.identity ? best.identity->confidenceThreshold : 0.65f);
    best.matched = (best.similarity >= thresh);
    return best;
}

void IdentityStore::RecordSighting(const std::string& id) {
    auto it = identities_.find(id);
    if (it == identities_.end()) return;
    it->second.encounterCount++;
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
    it->second.lastSeen = ss.str();
}

void IdentityStore::AddTag(const std::string& id, const std::string& tag) {
    auto it = identities_.find(id);
    if (it != identities_.end()) it->second.tags.push_back(tag);
}
void IdentityStore::AddRelationship(const std::string& id, const std::string& rel) {
    auto it = identities_.find(id);
    if (it != identities_.end()) it->second.relationships.push_back(rel);
}

void IdentityStore::Save() {
    nlohmann::json j = nlohmann::json::array();
    for (auto& [uid, ident] : identities_) {
        nlohmann::json entry;
        entry["id"]             = ident.id;
        entry["name"]           = ident.name;
        entry["embedding"]      = FaceEncoder::ToBase64(ident.embedding);
        entry["tags"]           = ident.tags;
        entry["relationships"]  = ident.relationships;
        entry["firstSeen"]      = ident.firstSeen;
        entry["lastSeen"]       = ident.lastSeen;
        entry["encounterCount"] = ident.encounterCount;
        entry["threshold"]      = ident.confidenceThreshold;
        j.push_back(entry);
    }
    // Ensure parent directory exists
    try {
        std::filesystem::path p(dbPath_);
        auto dir = p.parent_path();
        if (!dir.empty())
            std::filesystem::create_directories(dir);
    } catch (const std::exception& e) {
        std::cerr << "[IdentityStore] Save mkdir error: " << e.what() << "\n";
    }
    std::ofstream f(dbPath_);
    f << j.dump(2);
}

void IdentityStore::Load() {
    std::ifstream f(dbPath_);
    if (!f.is_open()) return;
    try {
        nlohmann::json j;
        f >> j;
        for (auto& entry : j) {
            Identity ident;
            ident.id             = entry["id"];
            ident.name           = entry["name"];
            ident.embedding      = FaceEncoder::FromBase64(entry["embedding"]);
            ident.tags           = entry["tags"].get<std::vector<std::string>>();
            ident.relationships  = entry["relationships"].get<std::vector<std::string>>();
            ident.firstSeen      = entry["firstSeen"];
            ident.lastSeen       = entry["lastSeen"];
            ident.encounterCount = entry["encounterCount"];
            ident.confidenceThreshold = entry["threshold"];
            identities_[ident.id] = ident;
        }
        std::cout << "[IdentityStore] Loaded " << identities_.size() << " identities\n";
    } catch (const std::exception& e) {
        std::cerr << "[IdentityStore] Load error: " << e.what() << "\n";
    }
}

std::vector<Identity> IdentityStore::GetAll() const {
    std::vector<Identity> out;
    for (auto& [k,v] : identities_) out.push_back(v);
    return out;
}

} // namespace hud
