/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/remote_callback.hpp>

TEST(RemoteCallbackTest, copy_nontemporal)
{
  // Sizes around the 32-byte vector width, and a full libcurl write callback (16 KiB) plus a tail.
  std::vector<std::size_t> const sizes{0, 1, 31, 32, 33, 16384 + 13};
  constexpr std::size_t max_size = 16384 + 13;

  std::vector<std::byte> src(max_size);
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::byte>(i * 7 + 1);
  }

  // Room for any misalignment in front of the destination and a margin behind it.
  alignas(32) std::array<std::byte, max_size + 64> buffer;
  std::array<std::byte, max_size + 64> expected;
  for (auto const size : sizes) {
    for (std::size_t misalignment = 0; misalignment < 32; ++misalignment) {
      buffer.fill(std::byte{0});
      expected.fill(std::byte{0});
      std::memcpy(expected.data() + misalignment, src.data(), size);

      kvikio::detail::copy_nontemporal(buffer.data() + misalignment, src.data(), size);

      // Also checks that no byte outside the destination was written.
      EXPECT_TRUE(buffer == expected) << "size " << size << ", misalignment " << misalignment;
    }
  }
}
