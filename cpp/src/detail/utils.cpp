/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

std::string normalize_env_value(std::string_view s)
{
  std::string str{s};
  std::transform(
    str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
  // `stringstream::operator>>` trims leading whitespace and stops at the next whitespace,
  // giving us the trimmed-and-lowercased value of the first token.
  std::stringstream trimmer;
  trimmer << str;
  str.clear();
  trimmer >> str;
  return str;
}

std::size_t align_up(std::size_t value, std::size_t alignment)
{
  KVIKIO_EXPECT((alignment > 0) && ((alignment & (alignment - 1)) == 0),
                "Alignment must be a power of 2");
  return (value + alignment - 1) & ~(alignment - 1);
}

void* align_up(void* addr, std::size_t alignment)
{
  KVIKIO_EXPECT((alignment > 0) && ((alignment & (alignment - 1)) == 0),
                "Alignment must be a power of 2");
  auto res = (reinterpret_cast<uintptr_t>(addr) + alignment - 1) & ~(alignment - 1);
  return reinterpret_cast<void*>(res);
}

std::size_t align_down(std::size_t value, std::size_t alignment)
{
  KVIKIO_EXPECT((alignment > 0) && ((alignment & (alignment - 1)) == 0),
                "Alignment must be a power of 2");
  return value & ~(alignment - 1);
}

void* align_down(void* addr, std::size_t alignment)
{
  KVIKIO_EXPECT((alignment > 0) && ((alignment & (alignment - 1)) == 0),
                "Alignment must be a power of 2");
  auto res = reinterpret_cast<uintptr_t>(addr) & ~(alignment - 1);
  return reinterpret_cast<void*>(res);
}

bool is_aligned(std::size_t value, std::size_t alignment)
{
  KVIKIO_EXPECT((alignment > 0) && ((alignment & (alignment - 1)) == 0),
                "Alignment must be a power of 2");
  return (value & (alignment - 1)) == 0;
}

bool is_aligned(void* addr, std::size_t alignment)
{
  KVIKIO_EXPECT((alignment > 0) && ((alignment & (alignment - 1)) == 0),
                "Alignment must be a power of 2");
  return (reinterpret_cast<uintptr_t>(addr) & (alignment - 1)) == 0;
}

void disable_read_optimization(void* addr)
{
  auto addr_byte = static_cast<std::byte*>(addr);
  std::byte tmp{};
  asm volatile("" : "+r,m"(tmp = *addr_byte) : : "memory");
}

std::ptrdiff_t pointer_diff(void* p1, void* p2)
{
  return static_cast<std::byte*>(p1) - static_cast<std::byte*>(p2);
}

std::size_t perform_prefault(void* buf, std::size_t size, bool write)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const page_size = get_page_size();
  auto aligned_addr    = detail::align_up(buf, page_size);
  auto const touch     = [write](void* addr) {
    if (write) {
      *static_cast<std::byte volatile*>(addr) = std::byte{};
    } else {
      detail::disable_read_optimization(addr);
    }
  };

  std::size_t touched_bytes{0};

  // If buf is not aligned, touch the byte at buf.
  auto num_bytes = detail::pointer_diff(aligned_addr, buf);
  if (num_bytes > 0) {
    touch(buf);
    touched_bytes += num_bytes;
    if (size >= num_bytes) { size -= num_bytes; }
  }

  if (num_bytes >= size) { return touched_bytes; }

  while (size > 0) {
    touch(aligned_addr);
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

}  // namespace kvikio::detail
