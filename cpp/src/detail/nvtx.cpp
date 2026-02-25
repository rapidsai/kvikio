/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/syscall.h>
#include <array>

#include <nvtx3/nvtx3.hpp>

#include <kvikio/detail/nvtx.hpp>

namespace kvikio {

NvtxCallTag::NvtxCallTag() : color(NvtxManager::default_color()) {}

NvtxCallTag::NvtxCallTag(std::uint64_t a_call_idx, NvtxColor a_color)
  : call_idx(a_call_idx), color(a_color)
{
}

NvtxManager& NvtxManager::instance() noexcept
{
  static NvtxManager _instance;
  return _instance;
}

const NvtxColor& NvtxManager::default_color() noexcept
{
  static NvtxColor default_color{nvtx3::argb{0, 255, 255, 255}};
  return default_color;
}

const NvtxColor& NvtxManager::get_color_by_index(std::uint64_t idx) noexcept
{
  constexpr std::size_t num_color{16};
  static_assert((num_color & (num_color - 1)) == 0);  // Is power of 2
  static std::array<NvtxColor, num_color> color_palette = {nvtx3::rgb{106, 192, 67},
                                                           nvtx3::rgb{191, 73, 203},
                                                           nvtx3::rgb{93, 151, 76},
                                                           nvtx3::rgb{96, 72, 194},
                                                           nvtx3::rgb{179, 170, 71},
                                                           nvtx3::rgb{92, 58, 113},
                                                           nvtx3::rgb{212, 136, 57},
                                                           nvtx3::rgb{96, 144, 194},
                                                           nvtx3::rgb{211, 69, 56},
                                                           nvtx3::rgb{97, 179, 155},
                                                           nvtx3::rgb{203, 69, 131},
                                                           nvtx3::rgb{57, 89, 48},
                                                           nvtx3::rgb{184, 133, 199},
                                                           nvtx3::rgb{128, 102, 51},
                                                           nvtx3::rgb{211, 138, 130},
                                                           nvtx3::rgb{122, 50, 49}};
  auto safe_idx                                         = idx & (num_color - 1);  // idx % num_color
  return color_palette[safe_idx];
}

NvtxCallTag NvtxManager::next_call_tag()
{
  static std::atomic_uint64_t call_counter{1ull};
  auto call_idx    = call_counter.fetch_add(1ull, std::memory_order_relaxed);
  auto& nvtx_color = NvtxManager::get_color_by_index(call_idx);
  return {call_idx, nvtx_color};
}

NvtxRegisteredString const& NvtxManager::get_empty_registered_string()
{
  static NvtxRegisteredString s("");
  return s;
}

}  // namespace kvikio
