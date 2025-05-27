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
#include <stdexcept>
#include <type_traits>

#include <kvikio/error.hpp>
#include <kvikio/mmap.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

namespace detail {

template <typename T>
void do_not_optimize_away_read(T const& value)
{
  asm volatile("" : : "r,m"(value) : "memory");
}

template <typename Integer>
void* pointer_add(void* p, Integer v)
{
  static_assert(std::is_integral_v<Integer>);
  return static_cast<std::byte*>(p) + v;
}

std::ptrdiff_t pointer_diff(void* p1, void* p2)
{
  return static_cast<std::byte*>(p1) - static_cast<std::byte*>(p2);
}
}  // namespace detail

MmapHandle::MmapHandle(std::string const& file_path,
                       std::string const& flags,
                       std::size_t size,
                       std::size_t file_offset,
                       void* external_buf,
                       mode_t mode)
  : _external_buf{external_buf},
    _size(size),
    _initial_file_offset(file_offset),
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
  // Case 1: External buffer is not specified
  //
  //     |--> file start                 |<--page_size-->|
  //     |
  // (0) |...............|...............|...............|...............|..........
  //
  // (1) |<-------_initial_file_offset-------->|<----_size----->|
  //                                           |--> _buf
  //
  // (2) |<---------_map_offset--------->|<------_map_size----->|
  //                                     |--> _map_addr
  //
  // (3) |<------------file_offset---------------->|<--size->|
  //                                           |--> _buf
  //
  // Case 2: External buffer is specified
  //
  //     |--> file start                 |<--page_size-->|
  //     |
  // (0) |...............|...............|...............|...............|..........
  //
  // (1) |<-------_initial_file_offset-------->|<------------_size------------->|
  //                                           |--> _buf/_external_buf
  //
  // (2) |<-----------------_map_offset----------------->|<------_map_size----->|
  //                                                     |--> _map_addr
  //
  // (3) |<------------file_offset---------------->|<--size->|
  //                                           |--> _buf/_external_buf

  auto const page_size = get_page_size();

  KVIKIO_EXPECT(
    _initial_file_offset < _file_size, "Offset is past the end of file", std::overflow_error);

  KVIKIO_EXPECT(_initial_file_offset + _size <= _file_size,
                "Mapped region is past the end of file",
                std::overflow_error);

  // Adjust _size to a valid value
  if (_size == 0 || (_initial_file_offset + _size) > _file_size) {
    _size = _file_size - _initial_file_offset;
  }

  if (!has_external_buf()) {
    // Case 1: External buffer is not specified
    _map_offset   = align_down(_initial_file_offset, page_size);
    _offset_delta = _initial_file_offset - _map_offset;
    _map_size     = _size + _offset_delta;
    _map_addr =
      mmap(nullptr, _map_size, _map_protection_flag, MAP_PRIVATE, _file_wrapper.fd(), _map_offset);
    _buf = detail::pointer_add(_map_addr, _offset_delta);
  } else {
    // Case 2: External buffer is specified
    _map_offset   = align_up(_initial_file_offset, page_size);
    _offset_delta = _map_offset - _initial_file_offset;

    if (_size <= _offset_delta) {
      _map_offset = 0;
      return;
    }

    _map_size = _size - _offset_delta;
    _buf      = _external_buf;
    auto res  = mmap(_map_addr,
                    _map_size,
                    _map_protection_flag,
                    MAP_PRIVATE | MAP_FIXED,
                    _file_wrapper.fd(),
                    _map_offset);
    KVIKIO_EXPECT(res == _map_addr, "Invalid mapped memory address");
  }

  SYSCALL_CHECK(_map_addr, "Cannot create memory mapping", MAP_FAILED);
}

void MmapHandle::unmap()
{
  if (_map_addr != nullptr) {
    auto ret = munmap(_map_addr, _map_size);
    SYSCALL_CHECK(ret);
  }
}

bool MmapHandle::has_external_buf() const noexcept { return _external_buf != nullptr; }

std::size_t MmapHandle::requested_size() const noexcept { return _size; }

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

std::pair<void*, std::size_t> MmapHandle::read(std::size_t size,
                                               std::size_t file_offset,
                                               bool prefault)
{
  KVIKIO_EXPECT(size > 0, "Read size must be greater than 0", std::invalid_argument);
  KVIKIO_EXPECT(
    file_offset >= _initial_file_offset && file_offset + size <= _initial_file_offset + _size,
    "Read is out of bound",
    std::invalid_argument);

  auto start_addr   = detail::pointer_add(_buf, file_offset - _initial_file_offset);
  auto aligned_addr = align_up(start_addr, get_page_size());

  if (has_external_buf() && _offset_delta > 0 && file_offset < _map_offset) {
    auto read_size = std::min(size, _map_offset - file_offset);
    detail::posix_host_io<detail::IOOperationType::READ, detail::PartialIO::NO>(
      _file_wrapper.fd(), start_addr, read_size, file_offset);
  }

  if (prefault) { perform_prefault(aligned_addr, size); }

  return {start_addr, size};
}

std::pair<void*, std::future<std::size_t>> MmapHandle::pread(std::size_t size,
                                                             std::size_t file_offset,
                                                             bool prefault)
{
  std::promise<std::size_t> p;
  return {nullptr, p.get_future()};
}

std::future<std::size_t> MmapHandle::pread(void* buf,
                                           std::size_t size,
                                           std::size_t file_offset,
                                           bool prefault)
{
  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(size, nvtx_color);
  if (!is_host_memory(buf)) { KVIKIO_FAIL("File-backed mapping for device is not supported yet."); }

  std::promise<std::size_t> p;
  return p.get_future();

  //   CUcontext ctx = get_context_from_pointer(buf);

  //   // Shortcut that circumvent the threadpool and use the POSIX backend directly.
  //   if (size < gds_threshold) {
  //     PushAndPopContext c(ctx);
  //     auto bytes_read = detail::posix_device_read(_file_direct_off.fd(), buf, size,
  //     file_offset, 0);
  //     // Maintain API consistency while making this trivial case synchronous.
  //     // The result in the future is immediately available after the call.
  //     return make_ready_future(bytes_read);
  //   }

  //   // Let's synchronize once instead of in each task.
  //   if (sync_default_stream && !get_compat_mode_manager().is_compat_mode_preferred()) {
  //     PushAndPopContext c(ctx);
  //     CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr));
  //   }

  //   // Regular case that use the threadpool and run the tasks in parallel
  //   auto task = [this, ctx](void* devPtr_base,
  //                           std::size_t size,
  //                           std::size_t file_offset,
  //                           std::size_t devPtr_offset) -> std::size_t {
  //     PushAndPopContext c(ctx);
  //     return read(devPtr_base, size, file_offset, devPtr_offset, /* sync_default_stream = */
  //     false);
  //   };
  //   auto [devPtr_base, base_size, devPtr_offset] = get_alloc_info(buf, &ctx);
  //   return parallel_io(
  //     task, devPtr_base, size, file_offset, task_size, devPtr_offset, call_idx, nvtx_color);
}

void MmapHandle::perform_prefault(void* buf, std::size_t size)
{
  auto const page_size = get_page_size();
  auto aligned_addr    = align_up(buf, page_size);

  // If buf is not aligned, read the byte at buf.
  auto num_bytes = detail::pointer_diff(aligned_addr, buf);
  if (num_bytes > 0) {
    detail::do_not_optimize_away_read(*static_cast<std::byte*>(buf));
    if (size >= num_bytes) { size -= num_bytes; }
  }

  while (size >= 0) {
    detail::do_not_optimize_away_read(*static_cast<std::byte*>(aligned_addr));
    if (size >= page_size) {
      aligned_addr = detail::pointer_add(aligned_addr, page_size);
      size -= page_size;
    } else {
      break;
    }
  }
}
}  // namespace kvikio
