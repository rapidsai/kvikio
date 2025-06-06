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
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include <kvikio/error.hpp>
#include <kvikio/mmap.hpp>
#include <kvikio/nvtx.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

namespace detail {

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
                       std::size_t initial_size,
                       std::size_t initial_file_offset,
                       mode_t mode)
  : _initial_size(initial_size),
    _initial_file_offset(initial_file_offset),
    _initialized{true},
    _map_core_flags{MAP_PRIVATE},
    _file_wrapper(file_path, flags, false /* o_direct */, mode)
{
  KVIKIO_NVTX_FUNC_RANGE();
  _file_size = get_file_size(_file_wrapper.fd());
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

  if (_file_size > 0) { map(); }
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
void MmapHandle::map()
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const page_size = get_page_size();

  KVIKIO_EXPECT(
    _initial_file_offset < _file_size, "Offset is past the end of file", std::overflow_error);

  // Adjust _initial_size to a valid value for the special case
  if (_initial_size == 0) { _initial_size = _file_size - _initial_file_offset; }

  KVIKIO_EXPECT(_initial_file_offset + _initial_size <= _file_size,
                "Mapped region is past the end of file",
                std::overflow_error);

  _map_offset             = align_down(_initial_file_offset, page_size);
  auto const offset_delta = _initial_file_offset - _map_offset;
  _map_size               = _initial_size + offset_delta;
  _map_addr               = mmap(
    nullptr, _map_size, _map_protection_flags, _map_core_flags, _file_wrapper.fd(), _map_offset);
  SYSCALL_CHECK(_map_addr, "Cannot create memory mapping", MAP_FAILED);
  _buf = detail::pointer_add(_map_addr, offset_delta);
}

void MmapHandle::unmap()
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_map_addr != nullptr) {
    auto ret = munmap(_map_addr, _map_size);
    SYSCALL_CHECK(ret);
  }
}

bool MmapHandle::closed() const noexcept { return !_initialized; }

void MmapHandle::close() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (closed()) { return; }
  try {
    unmap();
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

std::size_t MmapHandle::read(void* buf, std::size_t size, std::size_t file_offset)
{
  KVIKIO_NVTX_FUNC_RANGE();

  auto const is_buf_host_mem = is_host_memory(buf);
  CUcontext ctx{};
  if (!is_buf_host_mem) { ctx = get_context_from_pointer(buf); }

  auto const src_buf = detail::pointer_add(_buf, file_offset - _initial_file_offset);

  if (is_buf_host_mem) {
    std::memcpy(buf, src_buf, size);
  } else {
    PushAndPopContext c(ctx);
    CUstream stream = detail::StreamsByThread::get();
    CUDA_DRIVER_TRY(
      cudaAPI::instance().MemcpyHtoDAsync(convert_void2deviceptr(buf), src_buf, size, stream));
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
  }
  return size;
}

std::future<std::size_t> MmapHandle::pread(void* buf,
                                           std::size_t size,
                                           std::size_t file_offset,
                                           std::size_t task_size)
{
  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(size, nvtx_color);

  auto const is_buf_host_mem = is_host_memory(buf);
  CUcontext ctx{};
  if (!is_buf_host_mem) { ctx = get_context_from_pointer(buf); }

  auto const src_buf = detail::pointer_add(_buf, file_offset - _initial_file_offset);
  std::size_t actual_task_size =
    (task_size == 0) ? std::max<std::size_t>(1, size / defaults::num_threads()) : task_size;

  auto op = [global_src_buf = src_buf, is_buf_host_mem = is_buf_host_mem, ctx = ctx](
              void* buf, std::size_t size, std::size_t, std::size_t buf_offset) -> std::size_t {
    auto const src_buf = detail::pointer_add(global_src_buf, buf_offset);
    auto const dst_buf = detail::pointer_add(buf, buf_offset);

    if (is_buf_host_mem) {
      std::memcpy(dst_buf, src_buf, size);
    } else {
      PushAndPopContext c(ctx);
      CUstream stream = detail::StreamsByThread::get();
      CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
        convert_void2deviceptr(dst_buf), src_buf, size, stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
    }

    return size;
  };

  return parallel_io(
    op, buf, size, file_offset, actual_task_size, 0 /* global_buf_offset */, call_idx, nvtx_color);
}

}  // namespace kvikio
