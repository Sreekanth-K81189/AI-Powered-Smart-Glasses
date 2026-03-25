#pragma once
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

class TTSQueue {
public:
    // Push text — non-blocking, safe to call from any thread
    void push(const std::string& text);

    // Pop next text — BLOCKS until item available or stop() called
    // Returns false when stop() called and queue is empty
    bool waitAndPop(std::string& outText);

    // Unblock all waiting threads — call on shutdown
    void stop();

    // Reset stopped state — call before restarting
    void reset();

    // Discard all pending items
    void clear();

    size_t size() const;

private:
    mutable std::mutex          mtx_;
    std::condition_variable     cv_;
    std::queue<std::string>     queue_;
    bool                        stopped_ = false;
};

extern TTSQueue gTTSQueue;

