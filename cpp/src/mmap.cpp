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

// Case 1: External buffer is not specified
//
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
//
// Case 2: External buffer is specified
//
//     |--> file start                 |<--page_size-->|
//     |
// (0) |...............|...............|...............|...............|..........
//
// (1) |<-_initial_file_offset->|<---------------_initial_size--------------->|
//                              |--> _buf_external_buf
//
// (2) |<-------_map_offset----------->|<--------------_map_size------------->|
//                                     |--> _map_addr
//
// (3) |<-------file_offset-------->|<--size->|
//                             |--> _buf/_external_buf
//                                  |--> start_addr
//                     |--> start_aligned_addr
//
// (3) |<------------------------file_offset-------------------->|<--size->|
//                             |--> _buf/_external_buf
//                                                               |--> start_addr
//                                                     |--> start_aligned_addr

MmapHandle::MmapHandle(std::string const& file_path,
                       std::string const& flags,
                       std::size_t initial_size,
                       std::size_t initial_file_offset,
                       void* external_buf,
                       mode_t mode)
  : _external_buf{external_buf},
    _initial_size(initial_size),
    _initial_file_offset(initial_file_offset),
    _initialized{true},
    _file_wrapper(file_path, flags, false /* o_direct */, mode)
{
  KVIKIO_NVTX_FUNC_RANGE();
  _file_size = get_file_size(_file_wrapper.fd());
  switch (flags[0]) {
    case 'r': {
      _map_protection_flag = PROT_READ;
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
    _external_buf{std::exchange(o._external_buf, {})},
    _initial_size{std::exchange(o._initial_size, {})},
    _initial_file_offset{std::exchange(o._initial_file_offset, {})},
    _file_size{std::exchange(o._file_size, {})},
    _map_offset{std::exchange(o._map_offset, {})},
    _map_size{std::exchange(o._map_size, {})},
    _map_addr{std::exchange(o._map_addr, {})},
    _offset_delta{std::exchange(o._offset_delta, {})},
    _initialized{std::exchange(o._initialized, {})},
    _map_protection_flag{std::exchange(o._map_protection_flag, {})},
    _file_wrapper{std::exchange(o._file_wrapper, {})}
{
}

MmapHandle& MmapHandle::operator=(MmapHandle&& o) noexcept
{
  _buf                 = std::exchange(o._buf, {});
  _external_buf        = std::exchange(o._external_buf, {});
  _initial_size        = std::exchange(o._initial_size, {});
  _initial_file_offset = std::exchange(o._initial_file_offset, {});
  _file_size           = std::exchange(o._file_size, {});
  _map_offset          = std::exchange(o._map_offset, {});
  _map_size            = std::exchange(o._map_size, {});
  _map_addr            = std::exchange(o._map_addr, {});
  _offset_delta        = std::exchange(o._offset_delta, {});
  _initialized         = std::exchange(o._initialized, {});
  _map_protection_flag = std::exchange(o._map_protection_flag, {});
  _file_wrapper        = std::exchange(o._file_wrapper, {});

  return *this;
}

MmapHandle::~MmapHandle() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  close();
}

void MmapHandle::map()
{
  KVIKIO_NVTX_FUNC_RANGE();

  auto const page_size = get_page_size();

  KVIKIO_EXPECT(
    _initial_file_offset < _file_size, "Offset is past the end of file", std::overflow_error);

  KVIKIO_EXPECT(_initial_file_offset + _initial_size <= _file_size,
                "Mapped region is past the end of file",
                std::overflow_error);

  // Adjust _initial_size to a valid value
  if (_initial_size == 0 || (_initial_file_offset + _initial_size) > _file_size) {
    _initial_size = _file_size - _initial_file_offset;
  }

  if (!has_external_buf()) {
    // Case 1: External buffer is not specified
    _map_offset   = align_down(_initial_file_offset, page_size);
    _offset_delta = _initial_file_offset - _map_offset;
    _map_size     = _initial_size + _offset_delta;
    _map_addr =
      mmap(nullptr, _map_size, _map_protection_flag, MAP_PRIVATE, _file_wrapper.fd(), _map_offset);
    SYSCALL_CHECK(_map_addr, "Cannot create memory mapping", MAP_FAILED);
    _buf = detail::pointer_add(_map_addr, _offset_delta);
  } else {
    // Case 2: External buffer is specified
    _map_offset   = align_up(_initial_file_offset, page_size);
    _offset_delta = _map_offset - _initial_file_offset;

    if (_initial_size <= _offset_delta) {
      _map_offset = 0;
      _map_size   = 0;
      return;
    }

    _map_size = _initial_size - _offset_delta;
    _buf      = _external_buf;
    _map_addr = align_up(_buf, page_size);
    auto res  = mmap(_map_addr,
                    _map_size,
                    _map_protection_flag,
                    MAP_PRIVATE | MAP_FIXED_NOREPLACE,
                    _file_wrapper.fd(),
                    _map_offset);
    SYSCALL_CHECK(res, "Cannot create memory mapping", MAP_FAILED);
    KVIKIO_EXPECT(res == _map_addr, "Invalid mapped memory address");
  }
}

void MmapHandle::unmap()
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_map_addr != nullptr) {
    auto ret = munmap(_map_addr, _map_size);
    SYSCALL_CHECK(ret);
  }
}

bool MmapHandle::has_external_buf() const noexcept { return _external_buf != nullptr; }

std::size_t MmapHandle::requested_size() const noexcept { return _initial_size; }

bool MmapHandle::closed() const noexcept { return !_initialized; }

void MmapHandle::close() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (closed()) { return; }
  try {
    unmap();
  } catch (...) {
  }
  _file_wrapper.close();
  _initialized = false;
  _file_size   = 0;
  _map_size    = 0;
  _map_addr    = nullptr;
}

std::tuple<void*, void*, std::size_t, std::size_t> MmapHandle::prepare_read(std::size_t size,
                                                                            std::size_t file_offset)
{
  KVIKIO_EXPECT(size > 0, "Read size must be greater than 0", std::invalid_argument);
  KVIKIO_EXPECT(file_offset >= _initial_file_offset &&
                  file_offset + size <= _initial_file_offset + _initial_size,
                "Read is out of bound",
                std::invalid_argument);

  auto start_addr         = detail::pointer_add(_buf, file_offset - _initial_file_offset);
  auto start_aligned_addr = align_down(start_addr, get_page_size());
  auto adjusted_size      = detail::pointer_diff(start_addr, start_aligned_addr) + size;
  std::size_t posix_size{0};

  if (has_external_buf() && _offset_delta > 0 && file_offset < _map_offset) {
    auto read_size = std::min(size, _map_offset - file_offset);
    posix_size     = detail::posix_host_io<detail::IOOperationType::READ, detail::PartialIO::NO>(
      _file_wrapper.fd(), start_addr, read_size, file_offset);
  }

  return {start_addr, start_aligned_addr, adjusted_size, static_cast<std::size_t>(posix_size)};
}

std::pair<void*, std::size_t> MmapHandle::read(std::size_t size,
                                               std::size_t file_offset,
                                               bool prefault)
{
  KVIKIO_NVTX_FUNC_RANGE();

  auto const [start_addr, start_aligned_addr, adjusted_size, posix_size] =
    prepare_read(size, file_offset);

  std::size_t total_bytes{posix_size};

  if (prefault) {
    total_bytes += perform_prefault(start_aligned_addr, adjusted_size);
    total_bytes -= detail::pointer_diff(start_addr, start_aligned_addr);
  } else {
    total_bytes = size;
  }

  return {start_addr, total_bytes};
}

std::pair<void*, std::future<std::size_t>> MmapHandle::pread(std::size_t size,
                                                             std::size_t file_offset,
                                                             bool prefault,
                                                             std::size_t aligned_task_size)
{
  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(size, nvtx_color);

  auto const [start_addr, start_aligned_addr, adjusted_size, posix_size] =
    prepare_read(size, file_offset);

  if (prefault) {
    std::future<std::size_t> fut_prefault = perform_prefault_parallel(
      start_aligned_addr, adjusted_size, aligned_task_size, call_idx, nvtx_color);

    auto fut_gather = detail::submit_move_only_task(
      [fut_prefault       = std::move(fut_prefault),
       posix_size         = posix_size,
       start_addr         = start_addr,
       start_aligned_addr = start_aligned_addr]() mutable -> std::size_t {
        return posix_size + fut_prefault.get() -
               detail::pointer_diff(start_addr, start_aligned_addr);
      },
      call_idx,
      nvtx_color);

    return {start_addr, std::move(fut_gather)};
  } else {
    return {start_addr, make_ready_future(size)};
  }
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

std::future<std::size_t> MmapHandle::perform_prefault_parallel(void* buf,
                                                               std::size_t size,
                                                               std::size_t aligned_task_size,
                                                               std::uint64_t call_idx,
                                                               nvtx_color_type nvtx_color)
{
  KVIKIO_NVTX_FUNC_RANGE(size, nvtx_color);

  auto const page_size = get_page_size();

  KVIKIO_EXPECT((aligned_task_size & (page_size - 1)) == 0,
                "Task size must be a multiple of page size.",
                std::invalid_argument);

  auto aligned_addr = align_up(buf, page_size);
  std::size_t touched_bytes{0};

  // If buf is not aligned, read the byte at buf.
  auto num_bytes = detail::pointer_diff(aligned_addr, buf);
  if (num_bytes > 0) {
    detail::do_not_optimize_away_read(buf);
    touched_bytes += num_bytes;
    if (size >= num_bytes) { size -= num_bytes; }
  }

  auto op =
    [page_size = page_size](
      void* aligned_addr, std::size_t size, std::size_t, std::size_t buf_offset) -> std::size_t {
    aligned_addr = detail::pointer_add(aligned_addr, buf_offset);
    std::size_t touched_bytes{0};
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
  };

  auto fut = parallel_io(op, aligned_addr, size, 0, aligned_task_size, 0, call_idx, nvtx_color);

  return detail::submit_move_only_task(
    [fut = std::move(fut), touched_bytes = touched_bytes]() mutable -> std::size_t {
      return touched_bytes + fut.get();
    },
    call_idx,
    nvtx_color);
}

}  // namespace kvikio
