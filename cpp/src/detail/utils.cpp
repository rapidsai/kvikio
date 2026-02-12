/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {

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

}  // namespace kvikio::detail
