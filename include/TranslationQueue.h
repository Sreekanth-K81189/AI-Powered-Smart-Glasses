#pragma once
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>

// ── Translation job pushed by OCR or STT threads ────────────────────────────
struct TranslationJob {
    enum class Source { OCR, STT };

    Source      source;
    std::string text;          // raw text to translate
    std::string sourceLang;    // ISO 639-1 code: "ja", "zh", "fr", etc.
                               // set to "auto" if language detection not done yet
    std::string targetLang;    // from Config::targetLanguage at time of push
};

// ── Thread-safe blocking queue ───────────────────────────────────────────────
class TranslationQueue {
public:
    // Push a job — called from OCR or STT thread
    // Returns immediately, never blocks
    void push(TranslationJob job);

    // Pop a job — called from TranslationEngine thread only
    // BLOCKS until a job is available or stop() is called
    // Returns false if stop() was called and queue is empty (thread should exit)
    bool waitAndPop(TranslationJob& outJob);

    // Signal the queue to unblock any waiting thread (called on shutdown)
    void stop();

    // Returns number of pending jobs (for diagnostics/logging only)
    size_t size() const;

    // Discard all pending jobs (called on language change or reset)
    void clear();

private:
    mutable std::mutex              mtx_;
    std::condition_variable         cv_;
    std::queue<TranslationJob>      jobs_;
    bool                            stopped_ = false;
};

extern TranslationQueue gTranslationQueue;

