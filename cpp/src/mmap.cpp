/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <sys/mman.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <kvikio/error.hpp>
#include <kvikio/mmap.hpp>
#include <kvikio/nvtx.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/utils.hpp>
#include "kvikio/file_utils.hpp"

namespace kvikio {

namespace detail {
/**
 * @brief Prevent the compiler from optimizing away the read of a byte from a given address
 *
 * @param addr The address to read from
 */
void do_not_optimize_away_read(void* addr)
{
  auto addr_byte = static_cast<std::byte*>(addr);
  std::byte tmp{};
  asm volatile("" : "+r,m"(tmp = *addr_byte) : : "memory");
}

/**
 * @brief Change an address `p` by a signed difference of `v`
 *
 * @tparam Integer Signed integer type
 * @param p An address
 * @param v Change of `p` in bytes
 * @return A new address as a result of applying `v` on `p`
 *
 * @note This function exploits UB in C++.
 */
template <typename Integer>
void* pointer_add(void* p, Integer v)
{
  static_assert(std::is_integral_v<Integer>);
  return static_cast<std::byte*>(p) + v;
}

/**
 * @brief The distance in bytes between pointer `p1` and `p2`
 *
 * @param p1 The first pointer
 * @param p2 The second pointer
 * @return Signed result of (`p1` - `p2`). Both pointers are cast to std::byte* before subtraction.
 *
 * @note This function exploits UB in C++.
 */
std::ptrdiff_t pointer_diff(void* p1, void* p2)
{
  return static_cast<std::byte*>(p1) - static_cast<std::byte*>(p2);
}
}  // namespace detail

MmapHandle::MmapHandle(std::string const& file_path,
                       std::string const& flags,
                       std::optional<std::size_t> initial_size,
                       std::size_t initial_file_offset,
                       mode_t mode)
  : _initial_file_offset(initial_file_offset),
    _initialized{true},
    _map_core_flags{MAP_PRIVATE},
    _file_wrapper(file_path, flags, false /* o_direct */, mode)
{
  KVIKIO_NVTX_FUNC_RANGE();

  _file_size = get_file_size(_file_wrapper.fd());
  if (_file_size == 0) { return; }

  KVIKIO_EXPECT(
    _initial_file_offset < _file_size, "Offset is past the end of file", std::out_of_range);

  // An initial size of std::nullopt is a shorthand for "starting from _initial_file_offset to the
  // end of file".
  _initial_size =
    initial_size.has_value() ? initial_size.value() : (_file_size - _initial_file_offset);

  KVIKIO_EXPECT(_initial_size > 0, "Mapped region should not be zero byte", std::invalid_argument);
  KVIKIO_EXPECT(_initial_file_offset + _initial_size <= _file_size,
                "Mapped region is past the end of file",
                std::out_of_range);

  auto const page_size    = get_page_size();
  _map_offset             = align_down(_initial_file_offset, page_size);
  auto const offset_delta = _initial_file_offset - _map_offset;
  _map_size               = _initial_size + offset_delta;

  switch (flags[0]) {
    case 'r': {
      _map_protection_flags = PROT_READ;
      break;
    }
    case 'w': {
      KVIKIO_FAIL("File-backed mmap write is not supported yet", std::invalid_argument);
    }
    default: {
      KVIKIO_FAIL("Unknown file open flag", std::invalid_argument);
    }
  }

  _map_addr = mmap(
    nullptr, _map_size, _map_protection_flags, _map_core_flags, _file_wrapper.fd(), _map_offset);
  SYSCALL_CHECK(_map_addr, "Cannot create memory mapping", MAP_FAILED);
  _buf = detail::pointer_add(_map_addr, offset_delta);
}

MmapHandle::MmapHandle(MmapHandle&& o) noexcept
  : _buf{std::exchange(o._buf, {})},
    _initial_size{std::exchange(o._initial_size, {})},
    _initial_file_offset{std::exchange(o._initial_file_offset, {})},
    _file_size{std::exchange(o._file_size, {})},
    _map_offset{std::exchange(o._map_offset, {})},
    _map_size{std::exchange(o._map_size, {})},
    _map_addr{std::exchange(o._map_addr, {})},
    _initialized{std::exchange(o._initialized, {})},
    _map_protection_flags{std::exchange(o._map_protection_flags, {})},
    _map_core_flags{std::exchange(o._map_core_flags, {})},
    _file_wrapper{std::exchange(o._file_wrapper, {})}
{
}

MmapHandle& MmapHandle::operator=(MmapHandle&& o) noexcept
{
  _buf                  = std::exchange(o._buf, {});
  _initial_size         = std::exchange(o._initial_size, {});
  _initial_file_offset  = std::exchange(o._initial_file_offset, {});
  _file_size            = std::exchange(o._file_size, {});
  _map_offset           = std::exchange(o._map_offset, {});
  _map_size             = std::exchange(o._map_size, {});
  _map_addr             = std::exchange(o._map_addr, {});
  _initialized          = std::exchange(o._initialized, {});
  _map_protection_flags = std::exchange(o._map_protection_flags, {});
  _map_core_flags       = std::exchange(o._map_core_flags, {});
  _file_wrapper         = std::exchange(o._file_wrapper, {});

  return *this;
}

MmapHandle::~MmapHandle() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  close();
}

//     |--> file start                 |<--page_size-->|
//     |
// (0) |...............|...............|...............|...............|............
//
// (1) |<--_initial_file_offset-->|<---------------_initial_size--------------->|
//                                |--> _buf
//
// (2) |<-_map_offset->|<----------------------_map_size----------------------->|
//                     |--> _map_addr
//
// (3) |<---------------------file_offset--------------------->|<--size-->|
//                                |--> _buf
//                                                             |--> start_addr
//                                                     |--> start_aligned_addr

bool MmapHandle::closed() const noexcept { return !_initialized; }

void MmapHandle::close() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (closed() || _map_addr == nullptr) { return; }
  try {
    auto ret = munmap(_map_addr, _map_size);
    SYSCALL_CHECK(ret);
  } catch (...) {
  }
  _buf                  = {};
  _initial_size         = {};
  _initial_file_offset  = {};
  _file_size            = {};
  _map_offset           = {};
  _map_size             = {};
  _map_addr             = {};
  _initialized          = {};
  _map_protection_flags = {};
  _map_core_flags       = {};
  _file_wrapper         = {};
}

std::size_t MmapHandle::initial_size() const noexcept { return _initial_size; }

std::size_t MmapHandle::initial_file_offset() const noexcept { return _initial_file_offset; }

std::size_t MmapHandle::file_size() const
{
  if (closed()) { return 0; }
  return get_file_size(_file_wrapper.fd());
}

std::size_t MmapHandle::nbytes() const { return file_size(); }

std::size_t MmapHandle::read(void* buf, std::optional<std::size_t> size, std::size_t file_offset)
{
  KVIKIO_EXPECT(!closed(), "Cannot read from a closed MmapHandle", std::runtime_error);

  // Argument validation
  KVIKIO_EXPECT(file_offset < _file_size, "Offset is past the end of file", std::out_of_range);
  auto actual_size = size.has_value() ? size.value() : _file_size - file_offset;
  KVIKIO_EXPECT(actual_size > 0, "Read size must be greater than 0", std::invalid_argument);
  KVIKIO_EXPECT(file_offset >= _initial_file_offset &&
                  file_offset + actual_size <= _initial_file_offset + _initial_size,
                "Read is out of bound",
                std::out_of_range);

  KVIKIO_NVTX_FUNC_RANGE();

  auto const is_buf_host_mem = is_host_memory(buf);
  CUcontext ctx{};
  if (!is_buf_host_mem) { ctx = get_context_from_pointer(buf); }

  auto const src_buf = detail::pointer_add(_buf, file_offset - _initial_file_offset);

  if (is_buf_host_mem) {
    std::memcpy(buf, src_buf, actual_size);
  } else {
    PushAndPopContext c(ctx);
    CUstream stream = detail::StreamsByThread::get();
    CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
      convert_void2deviceptr(buf), src_buf, actual_size, stream));
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
  }
  return actual_size;
}

std::future<std::size_t> MmapHandle::pread(void* buf,
                                           std::optional<std::size_t> size,
                                           std::size_t file_offset,
                                           std::size_t mmap_task_size)
{
  KVIKIO_EXPECT(!closed(), "Cannot read from a closed MmapHandle", std::runtime_error);

  // Argument validation
  KVIKIO_EXPECT(file_offset < _file_size, "Offset is past the end of file", std::out_of_range);
  auto actual_size = size.has_value() ? size.value() : _file_size - file_offset;
  KVIKIO_EXPECT(actual_size > 0, "Read size must be greater than 0", std::invalid_argument);
  KVIKIO_EXPECT(file_offset >= _initial_file_offset &&
                  file_offset + actual_size <= _initial_file_offset + _initial_size,
                "Read is out of bound",
                std::out_of_range);

  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(actual_size, nvtx_color);

  auto const is_buf_host_mem = is_host_memory(buf);
  CUcontext ctx{};
  if (!is_buf_host_mem) { ctx = get_context_from_pointer(buf); }

  auto const src_buf = detail::pointer_add(_buf, file_offset - _initial_file_offset);
  std::size_t actual_mmap_task_size =
    (mmap_task_size == 0) ? std::max<std::size_t>(1, actual_size / defaults::num_threads())
                          : mmap_task_size;

  auto op = [this, global_src_buf = src_buf, is_buf_host_mem = is_buf_host_mem, ctx = ctx](
              void* buf, std::size_t size, std::size_t, std::size_t buf_offset) -> std::size_t {
    auto const src_buf = detail::pointer_add(global_src_buf, buf_offset);
    auto const dst_buf = detail::pointer_add(buf, buf_offset);

    if (is_buf_host_mem) {
      std::memcpy(dst_buf, src_buf, size);
    } else {
      perform_prefault(src_buf, size);
      PushAndPopContext c(ctx);
      CUstream stream = detail::StreamsByThread::get();
      CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
        convert_void2deviceptr(dst_buf), src_buf, size, stream));
    }

    return size;
  };

  auto last_task_callback = [is_buf_host_mem = is_buf_host_mem, ctx = ctx] {
    if (!is_buf_host_mem) {
      PushAndPopContext c(ctx);
      CUstream stream = detail::StreamsByThread::get();
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
    }
  };

  return parallel_io(op,
                     buf,
                     actual_size,
                     file_offset,
                     actual_mmap_task_size,
                     0 /* global_buf_offset */,
                     call_idx,
                     nvtx_color,
                     last_task_callback);
}

std::size_t MmapHandle::perform_prefault(void* buf, std::size_t size)
{
  auto const page_size = get_page_size();
  auto aligned_addr    = align_up(buf, page_size);

  std::size_t touched_bytes{0};

  // If buf is not aligned, read the byte at buf.
  auto num_bytes = detail::pointer_diff(aligned_addr, buf);
  if (num_bytes > 0) {
    detail::do_not_optimize_away_read(buf);
    touched_bytes += num_bytes;
    if (size >= num_bytes) { size -= num_bytes; }
  }

  if (num_bytes >= size) { return touched_bytes; }

  while (size > 0) {
    detail::do_not_optimize_away_read(aligned_addr);
    if (size >= page_size) {
      aligned_addr = detail::pointer_add(aligned_addr, page_size);
      size -= page_size;
      touched_bytes += page_size;
    } else {
      touched_bytes += size;
      break;
    }
  }
  return touched_bytes;
}

}  // namespace kvikio
