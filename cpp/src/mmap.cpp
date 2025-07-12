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
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

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
 * @note It is UB in C++ when the initial pointer and the resulting pointer do not point to the
 * elements from the same array. This UB is considered acceptable here due to lack of a better
 * alternative.
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
 * @note It is UB in C++ when the two pointers engaged in subtraction do not point to the elements
 * from the same array. This UB is considered acceptable here due to lack of a better alternative.
 */
std::ptrdiff_t pointer_diff(void* p1, void* p2)
{
  return static_cast<std::byte*>(p1) - static_cast<std::byte*>(p2);
}

/**
 * @brief Whether the current device supports address translation service (ATS), whereby the CPU and
 * GPU share a single page table.
 *
 * @return Boolean answer
 */
bool is_ats_available()
{
  // Memoize the ATS availability record of all devices
  static auto ats_availability = []() -> auto {
    std::unordered_map<CUdevice, int> result;
    int num_devices{};
    CUDA_DRIVER_TRY(cudaAPI::instance().DeviceGetCount(&num_devices));
    for (int device_ordinal = 0; device_ordinal < num_devices; ++device_ordinal) {
      CUdevice device_handle{};
      CUDA_DRIVER_TRY(cudaAPI::instance().DeviceGet(&device_handle, device_ordinal));
      int attr{};
      CUDA_DRIVER_TRY(cudaAPI::instance().DeviceGetAttribute(
        &attr,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
        device_handle));
      result[device_handle] = attr;
    }
    return result;
  }();

  // Get current device
  CUdevice device_handle{};
  CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetDevice(&device_handle));

  // Look up the record
  return ats_availability[device_handle];
}

}  // namespace detail

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
// (0): Layout of the file-backed memory mapping if the whole file were mapped
// (1): At mapping handle construction time, the member `_initial_file_offset` and `_initial_size`
// determine the mapped region (2): `_map_addr` is the page aligned address returned by `mmap`.
// `_map_offset` is the adjusted offset
// (3): At read time, the argument `file_offset` and `size` determine the region to be read. This
// region must be a subset of the one defined at mapping handle construction time
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
  close();
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
  KVIKIO_NVTX_FUNC_RANGE();

  auto actual_size = validate_and_adjust_read_args(size, file_offset);

  auto const is_dst_buf_host_mem = is_host_memory(buf);
  CUcontext ctx{};
  if (!is_dst_buf_host_mem) { ctx = get_context_from_pointer(buf); }

  // Copy `actual_size` bytes from `src_mapped_buf` (src) to `buf` (dst)
  auto const src_mapped_buf = detail::pointer_add(_buf, file_offset - _initial_file_offset);
  read_impl(buf, src_mapped_buf, actual_size, 0, is_dst_buf_host_mem, ctx);
  return actual_size;
}

std::future<std::size_t> MmapHandle::pread(void* buf,
                                           std::optional<std::size_t> size,
                                           std::size_t file_offset,
                                           std::size_t mmap_task_size)
{
  KVIKIO_EXPECT(mmap_task_size <= defaults::bounce_buffer_size(),
                "bounce buffer size cannot be less than mmap_task_size.");
  auto actual_size = validate_and_adjust_read_args(size, file_offset);

  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(actual_size, nvtx_color);

  auto const is_dst_buf_host_mem = is_host_memory(buf);
  CUcontext ctx{};
  if (!is_dst_buf_host_mem) { ctx = get_context_from_pointer(buf); }

  // Copy `actual_size` bytes from `src_mapped_buf` (src) to `buf` (dst)
  auto const src_mapped_buf = detail::pointer_add(_buf, file_offset - _initial_file_offset);
  auto op =
    [this, src_mapped_buf = src_mapped_buf, is_dst_buf_host_mem = is_dst_buf_host_mem, ctx = ctx](
      void* dst_buf,
      std::size_t size,
      std::size_t,  // file_offset will be taken into account by dst_buf, hence no longer used here
      std::size_t buf_offset  // buf_offset will be incremented for each individual task
      ) -> std::size_t {
    read_impl(dst_buf, src_mapped_buf, size, buf_offset, is_dst_buf_host_mem, ctx);
    return size;
  };

  return parallel_io(op,
                     buf,
                     actual_size,
                     file_offset,
                     mmap_task_size,
                     0,  // dst buffer offset initial value
                     call_idx,
                     nvtx_color);
}

std::size_t MmapHandle::validate_and_adjust_read_args(std::optional<std::size_t> const& size,
                                                      std::size_t& file_offset)
{
  KVIKIO_EXPECT(!closed(), "Cannot read from a closed MmapHandle", std::runtime_error);
  KVIKIO_EXPECT(file_offset < _file_size, "Offset is past the end of file", std::out_of_range);
  auto actual_size = size.has_value() ? size.value() : _file_size - file_offset;
  KVIKIO_EXPECT(actual_size > 0, "Read size must be greater than 0", std::invalid_argument);
  KVIKIO_EXPECT(file_offset >= _initial_file_offset &&
                  file_offset + actual_size <= _initial_file_offset + _initial_size,
                "Read is out of bound",
                std::out_of_range);
  return actual_size;
}

void MmapHandle::read_impl(void* dst_buf,
                           void* src_mapped_buf,
                           std::size_t size,
                           std::size_t buf_offset,
                           bool is_dst_buf_host_mem,
                           CUcontext ctx)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const src = detail::pointer_add(src_mapped_buf, buf_offset);
  auto const dst = detail::pointer_add(dst_buf, buf_offset);

  if (is_dst_buf_host_mem) {
    // std::memcpy implicitly performs prefault for the mapped memory.
    std::memcpy(dst, src, size);
    return;
  }

  // Empirically, take the following steps to achieve good performance:
  // - On C2C:
  //   - Explicitly prefault
  //   - Copy from the mapped memory (pageable) to the device buffer
  // - On PCIe:
  //   - std::memcpy from the mapped memory to the pinned bounce buffer (which implicitly
  //     prefaults)
  //   - Copy from the bounce buffer to the device buffer

  PushAndPopContext c(ctx);
  CUstream stream = detail::StreamsByThread::get();

  auto h2d_batch_cpy_sync =
    [](CUdeviceptr dst_devptr, CUdeviceptr src_devptr, std::size_t size, CUstream stream) {
#if CUDA_VERSION >= 12080
      if (cudaAPI::instance().MemcpyBatchAsync) {
        CUmemcpyAttributes attrs{};
        std::size_t attrs_idxs[] = {0};
        attrs.srcAccessOrder     = CUmemcpySrcAccessOrder_enum::CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
        CUDA_DRIVER_TRY(
          cudaAPI::instance().MemcpyBatchAsync(&dst_devptr,
                                               &src_devptr,
                                               &size,
                                               static_cast<std::size_t>(1) /* count */,
                                               &attrs,
                                               attrs_idxs,
                                               static_cast<std::size_t>(1) /* num_attrs */,
                                               static_cast<std::size_t*>(nullptr),
                                               stream));
      } else {
        // Fall back to the conventional H2D copy if the batch copy API is not available.
        CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
          dst_devptr, reinterpret_cast<void*>(src_devptr), size, stream));
      }
#else
      CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
        dst_devptr, reinterpret_cast<void*>(src_devptr), size, stream));
#endif
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
    };

  auto dst_devptr = convert_void2deviceptr(dst);
  CUdeviceptr src_devptr{};
  if (detail::is_ats_available()) {
    perform_prefault(src, size);
    src_devptr = convert_void2deviceptr(src);
    h2d_batch_cpy_sync(dst_devptr, src_devptr, size, stream);
  } else {
    auto alloc = AllocRetain::instance().get();
    std::memcpy(alloc.get(), src, size);
    src_devptr = convert_void2deviceptr(alloc.get());
    h2d_batch_cpy_sync(dst_devptr, src_devptr, size, stream);
  }
}

std::size_t MmapHandle::perform_prefault(void* buf, std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
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
