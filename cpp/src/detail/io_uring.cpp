/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/io_uring.hpp>
#include <kvikio/error.hpp>
#include <kvikio/utils.hpp>
#include <memory>

#define IO_URING_CHECK(err_code)                                       \
  do {                                                                 \
    kvikio::detail::check_io_uring_call(__LINE__, __FILE__, err_code); \
  } while (0)

namespace kvikio::detail {
namespace {
inline void check_io_uring_call(int line_number, char const* filename, int err_code)
{
  // Success
  if (err_code == 0) { return; }

  // On failure, io_uring API returns -errno
  std::stringstream ss;
  ss << "Linux io_uring error (" << err_code << ") at: " << filename << ":" << line_number;

  // std::system_error::what() automatically contains the detailed error description
  // equivalent to calling strerrordesc_np(errno)
  throw kvikio::GenericSystemError(ss.str());
}
}  // namespace

IoUringManager::IoUringManager() : _queue_depth{32}, _task_size{defaults::task_size()}
{
  IO_URING_CHECK(io_uring_queue_init(_queue_depth, &_ring, 0));
}

IoUringManager& IoUringManager::get()
{
  static IoUringManager inst;
  return inst;
}

IoUringManager::~IoUringManager() noexcept
{
  // Does not have a return value
  io_uring_queue_exit(&_ring);
}

io_uring* IoUringManager::ring() noexcept { return &_ring; }

unsigned int IoUringManager::queue_depth() noexcept { return _queue_depth; }

std::size_t IoUringManager::task_size() noexcept { return _task_size; }

bool is_io_uring_supported() noexcept
{
  try {
    [[maybe_unused]] auto& inst = IoUringManager::get();
  } catch (...) {
    return false;
  }
  return true;
}

std::size_t io_uring_read_host(int fd, void* buf, std::size_t size, std::size_t file_offset)
{
  auto* ring       = IoUringManager::get().ring();
  auto queue_depth = IoUringManager::get().queue_depth();
  auto task_size   = IoUringManager::get().task_size();

  std::size_t inflight_queues{0};
  std::size_t bytes_submitted{0};
  std::size_t bytes_completed{0};
  std::size_t current_offset{file_offset};

  while (bytes_completed < size) {
    while (inflight_queues < queue_depth && bytes_submitted < size) {
      auto* sqe = io_uring_get_sqe(ring);

      // Queue is full. Need to consume some CQEs.
      if (sqe == nullptr) { break; }

      auto current_task_size = std::min(task_size, size - bytes_submitted);

      io_uring_prep_read(
        sqe, fd, static_cast<std::byte*>(buf) + bytes_submitted, current_task_size, current_offset);
      // Ask the kernel to execute the SQE operation asynchronously
      sqe->flags |= IOSQE_ASYNC;

      bytes_submitted += current_task_size;
      current_offset += current_task_size;
      ++inflight_queues;
    }

    auto num_sqes_submitted = io_uring_submit(ring);
    if (num_sqes_submitted < 0) { IO_URING_CHECK(num_sqes_submitted); }

    // Wait for one completion event
    struct io_uring_cqe* cqe{};
    IO_URING_CHECK(io_uring_wait_cqe(ring, &cqe));

    // Process all completion events at this point
    unsigned int head{0};
    unsigned int num_consumed{0};
    io_uring_for_each_cqe(ring, head, cqe)
    {
      if (cqe->res < 0) { IO_URING_CHECK(cqe->res); }
      bytes_completed += cqe->res;
      --inflight_queues;
      ++num_consumed;
    }

    // Mark completion events as consumed
    io_uring_cq_advance(ring, num_consumed);
  }

  KVIKIO_EXPECT(bytes_completed == bytes_submitted,
                "Loss of data: submission and completion mismatch.");

  return bytes_completed;
}

struct IoUringTaskCtx {
  void* bounce_buffer;
  void* src{};
  void* dst{};
  std::size_t size{};
};

std::stack<IoUringTaskCtx*>& task_ctx_pool()
{
  static auto task_ctx_objs = []() {
    std::vector<IoUringTaskCtx> result(IoUringManager::get().queue_depth());
    for (auto&& task_ctx : result) {
      void* buffer{};
      CUDA_DRIVER_TRY(cudaAPI::instance().MemHostAlloc(
        &buffer, defaults::bounce_buffer_size(), CU_MEMHOSTALLOC_PORTABLE));
      task_ctx.bounce_buffer = buffer;
    }
    return result;
  }();

  static auto task_ctx_pool = [&]() {
    std::stack<IoUringTaskCtx*> result;
    for (auto&& task_ctx : task_ctx_objs) {
      result.push(&task_ctx);
    }
    return result;
  }();
  return task_ctx_pool;
}

std::size_t io_uring_read_device(
  int fd, void* buf, std::size_t size, std::size_t file_offset, CUstream stream)
{
  KVIKIO_EXPECT(
    defaults::bounce_buffer_size() >= defaults::task_size(),
    "KvikIO requires that the bounce buffer size be no less than the task size for io_uring.");

  auto* ring       = IoUringManager::get().ring();
  auto queue_depth = IoUringManager::get().queue_depth();
  auto task_size   = IoUringManager::get().task_size();

  std::size_t inflight_queues{0};
  std::size_t bytes_submitted{0};
  std::size_t bytes_completed{0};
  std::size_t current_offset{file_offset};

  while (bytes_completed < size) {
    while (inflight_queues < queue_depth && bytes_submitted < size) {
      auto* sqe = io_uring_get_sqe(ring);

      // Queue is full. Need to consume some CQEs.
      if (sqe == nullptr) { break; }

      auto current_task_size = std::min(task_size, size - bytes_submitted);

      auto* task_ctx = task_ctx_pool().top();
      KVIKIO_EXPECT(!task_ctx_pool().empty(), "Task context pool is empty unexpectedly.");
      task_ctx_pool().pop();
      task_ctx->src  = task_ctx->bounce_buffer;
      task_ctx->dst  = static_cast<std::byte*>(buf) + bytes_submitted;
      task_ctx->size = current_task_size;

      io_uring_prep_read(sqe, fd, task_ctx->bounce_buffer, current_task_size, current_offset);
      // Ask the kernel to execute the SQE operation asynchronously
      sqe->flags |= IOSQE_ASYNC;

      io_uring_sqe_set_data(sqe, task_ctx);

      bytes_submitted += current_task_size;
      current_offset += current_task_size;
      ++inflight_queues;
    }

    auto num_sqes_submitted = io_uring_submit(ring);
    if (num_sqes_submitted < 0) { IO_URING_CHECK(num_sqes_submitted); }

    // Wait for one completion event
    struct io_uring_cqe* cqe{};
    IO_URING_CHECK(io_uring_wait_cqe(ring, &cqe));

    // Process all completion events at this point
    unsigned int head{0};
    unsigned int num_consumed{0};
    io_uring_for_each_cqe(ring, head, cqe)
    {
      if (cqe->res < 0) { IO_URING_CHECK(cqe->res); }
      bytes_completed += cqe->res;
      --inflight_queues;
      ++num_consumed;
      auto* task_ctx = static_cast<IoUringTaskCtx*>(io_uring_cqe_get_data(cqe));
      CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
        convert_void2deviceptr(task_ctx->dst), task_ctx->src, task_ctx->size, stream));
      task_ctx_pool().push(task_ctx);
    }

    // Mark completion events as consumed
    io_uring_cq_advance(ring, num_consumed);
  }

  CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));

  KVIKIO_EXPECT(bytes_completed == bytes_submitted,
                "Loss of data: submission and completion mismatch.");

  return bytes_completed;
}

}  // namespace kvikio::detail
