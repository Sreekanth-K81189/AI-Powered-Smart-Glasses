#pragma once
#include "face/face_encoder.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <nlohmann/json.hpp>

namespace hud {

struct Identity {
    std::string id;          // UUID
    std::string name;
    Embedding   embedding;   // 512-d
    std::vector<std::string> tags;          // ["family","staff","vip"]
    std::vector<std::string> relationships; // ["friend:uuid2","spouse:uuid3"]
    std::string firstSeen;
    std::string lastSeen;
    int         encounterCount = 0;
    float       confidenceThreshold = 0.65f;
};

struct MatchResult {
    std::optional<Identity> identity;
    float similarity = 0.0f;
    bool  matched    = false;
};

class IdentityStore {
public:
    explicit IdentityStore(const std::string& dbPath);
    ~IdentityStore();

    // Add new identity from multiple face images
    std::string AddIdentity(const std::string& name,
                            const Embedding& embedding,
                            const std::vector<std::string>& tags = {},
                            float threshold = 0.65f);

    // Find best match for a query embedding
    MatchResult FindMatch(const Embedding& query, float threshold = 0.0f);

    // Update last seen + counter
    void RecordSighting(const std::string& id);

    // Tag management
    void AddTag(const std::string& id, const std::string& tag);
    void AddRelationship(const std::string& id, const std::string& rel);

    // Persistence
    void Save();
    void Load();

    size_t Count() const { return identities_.size(); }
    std::vector<Identity> GetAll() const;

private:
    std::string dbPath_;
    std::unordered_map<std::string, Identity> identities_;
    std::string GenerateUUID();
};

} // namespace hud
