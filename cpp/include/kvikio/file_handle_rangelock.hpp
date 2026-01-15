/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Modified FileHandle with range-based locking support
 */
#pragma once

#include <kvikio/file_handle.hpp>
#include <kvikio/range_lock.hpp>
#include <memory>

namespace kvikio {

class FileHandleWithRangeLock : public FileHandle {
private:
    mutable RangeLockManager range_lock_manager_;

public:
    using FileHandle::FileHandle;  // Inherit constructors

    /**
     * @brief Write with range-based locking
     *
     * This version acquires a lock only for the specific range being written,
     * allowing non-overlapping writes to proceed in parallel.
     */
    std::future<std::size_t> pwrite_rangelock(void const* buf,
                                              std::size_t size,
                                              std::size_t file_offset = 0,
                                              std::size_t task_size = defaults::task_size(),
                                              std::size_t gds_threshold = defaults::gds_threshold(),
                                              bool sync_default_stream = true) {

        // Acquire range lock for this write
        auto range_lock = range_lock_manager_.lock_range(file_offset, file_offset + size);

        // Perform the write using the base class implementation
        auto future = this->pwrite(buf, size, file_offset, task_size, gds_threshold, sync_default_stream);

        // Create a wrapper future that releases the lock when done
        return std::async(std::launch::deferred, [future = std::move(future),
                                                  lock = std::move(range_lock)]() mutable {
            auto result = future.get();
            // Lock will be automatically released when this lambda exits
            return result;
        });
    }

    /**
     * @brief Read with range-based locking (optional, for consistency)
     */
    std::future<std::size_t> pread_rangelock(void* buf,
                                             std::size_t size,
                                             std::size_t file_offset = 0,
                                             std::size_t task_size = defaults::task_size(),
                                             std::size_t gds_threshold = defaults::gds_threshold(),
                                             bool sync_default_stream = true) {

        // For reads, we could use shared locks if needed
        // For now, using exclusive locks for simplicity
        auto range_lock = range_lock_manager_.lock_range(file_offset, file_offset + size);

        auto future = this->pread(buf, size, file_offset, task_size, gds_threshold, sync_default_stream);

        return std::async(std::launch::deferred, [future = std::move(future),
                                                  lock = std::move(range_lock)]() mutable {
            auto result = future.get();
            return result;
        });
    }

    /**
     * @brief Check if a range is currently locked
     */
    bool is_range_locked(std::size_t start, std::size_t end) const {
        return range_lock_manager_.is_range_locked(start, end);
    }

    /**
     * @brief Get statistics about locked ranges
     */
    std::size_t num_locked_ranges() const {
        return range_lock_manager_.num_locked_ranges();
    }
};

} // namespace kvikio