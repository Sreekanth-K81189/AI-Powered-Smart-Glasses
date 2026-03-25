#include "TranslationQueue.h"

void TranslationQueue::push(TranslationJob job) {
    std::lock_guard<std::mutex> lock(mtx_);
    jobs_.push(std::move(job));
    cv_.notify_one();
}

bool TranslationQueue::waitAndPop(TranslationJob& outJob) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]{ return !jobs_.empty() || stopped_; });
    if (stopped_ && jobs_.empty()) return false;
    outJob = std::move(jobs_.front());
    jobs_.pop();
    return true;
}

void TranslationQueue::stop() {
    std::lock_guard<std::mutex> lock(mtx_);
    stopped_ = true;
    cv_.notify_all();
}

size_t TranslationQueue::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return jobs_.size();
}

void TranslationQueue::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    jobs_ = std::queue<TranslationJob>{};
}

TranslationQueue gTranslationQueue;

