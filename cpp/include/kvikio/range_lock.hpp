/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Range-based locking for parallel file I/O
 * This allows non-overlapping ranges to be written in parallel
 */
#pragma once

#include <map>
#include <mutex>
#include <condition_variable>
#include <set>
#include <memory>

namespace kvikio {

class RangeLockManager {
public:
    struct Range {
        std::size_t start;
        std::size_t end;

        bool overlaps(const Range& other) const {
            return !(end <= other.start || start >= other.end);
        }

        bool operator<(const Range& other) const {
            return start < other.start;
        }
    };

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::set<Range> locked_ranges_;

public:
    class RangeLock {
    private:
        RangeLockManager* manager_;
        Range range_;
        bool locked_;

    public:
        RangeLock(RangeLockManager* manager, std::size_t start, std::size_t end)
            : manager_(manager), range_{start, end}, locked_(false) {
            lock();
        }

        ~RangeLock() {
            if (locked_) {
                unlock();
            }
        }

        // Move only
        RangeLock(const RangeLock&) = delete;
        RangeLock& operator=(const RangeLock&) = delete;
        RangeLock(RangeLock&& other) noexcept
            : manager_(other.manager_), range_(other.range_), locked_(other.locked_) {
            other.locked_ = false;
        }

        void lock() {
            if (locked_) return;

            std::unique_lock<std::mutex> lock(manager_->mutex_);

            // Wait until no overlapping ranges are locked
            manager_->cv_.wait(lock, [this]() {
                for (const auto& locked_range : manager_->locked_ranges_) {
                    if (range_.overlaps(locked_range)) {
                        return false;
                    }
                }
                return true;
            });

            // Lock this range
            manager_->locked_ranges_.insert(range_);
            locked_ = true;
        }

        void unlock() {
            if (!locked_) return;

            std::unique_lock<std::mutex> lock(manager_->mutex_);
            manager_->locked_ranges_.erase(range_);
            locked_ = false;

            // Notify waiting threads
            manager_->cv_.notify_all();
        }
    };

    std::unique_ptr<RangeLock> lock_range(std::size_t start, std::size_t end) {
        return std::make_unique<RangeLock>(this, start, end);
    }

    // Check if a range is currently locked
    bool is_range_locked(std::size_t start, std::size_t end) const {
        std::unique_lock<std::mutex> lock(mutex_);
        Range query{start, end};
        for (const auto& locked_range : locked_ranges_) {
            if (query.overlaps(locked_range)) {
                return true;
            }
        }
        return false;
    }

    // Get number of currently locked ranges
    std::size_t num_locked_ranges() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return locked_ranges_.size();
    }
};

} // namespace kvikio