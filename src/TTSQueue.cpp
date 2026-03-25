#include "TTSQueue.h"

TTSQueue gTTSQueue;

void TTSQueue::push(const std::string& text) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (stopped_) return;
        queue_.push(text);
    }
    cv_.notify_one();
}

bool TTSQueue::waitAndPop(std::string& outText) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]{ return !queue_.empty() || stopped_; });
    if (stopped_ && queue_.empty()) return false;
    outText = std::move(queue_.front());
    queue_.pop();
    return true;
}

void TTSQueue::stop() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stopped_ = true;
    }
    cv_.notify_all();
}

void TTSQueue::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    stopped_ = false;
}

void TTSQueue::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_ = std::queue<std::string>{};
}

size_t TTSQueue::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
}
