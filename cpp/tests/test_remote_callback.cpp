/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <vector>

#include <curl/curl.h>
#include <gmock/gmock.h>

#include <kvikio/detail/remote_callback.hpp>

using kvikio::detail::callback_host_memory;
using kvikio::detail::CallbackContext;
using kvikio::detail::TransferSegment;

namespace {

// The span of the running example in transfer_plan.hpp: three wanted pieces separated by two
// gaps.
//
//   span      0        60      110     140     190     240
//             |########|~~~~~~~~|######|~~~~~~~|######|
//                 S0      gap      S1    gap     S2
constexpr std::size_t span_size = 240;

class RemoteCallbackTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    for (std::size_t i = 0; i < span_size; ++i) {
      _source[i] = static_cast<std::byte>(i % 251);
    }
    _ctx.size = span_size;
    _ctx.segments.push_back({.span_offset = 0, .length = 60, .buf = _dst0.data()});
    _ctx.segments.push_back({.span_offset = 110, .length = 30, .buf = _dst1.data()});
    _ctx.segments.push_back({.span_offset = 190, .length = 50, .buf = _dst2.data()});
  }

  // Hands `nbytes` of the span to the callback, starting where the last call stopped.
  std::size_t feed(std::size_t nbytes)
  {
    auto* from = reinterpret_cast<char*>(_source.data()) + _fed;
    _fed += nbytes;
    return callback_host_memory(from, 1, nbytes, &_ctx);
  }

  // The bytes of the span that a segment should have received.
  [[nodiscard]] std::vector<std::byte> expected(std::size_t span_offset, std::size_t length) const
  {
    return {_source.begin() + static_cast<std::ptrdiff_t>(span_offset),
            _source.begin() + static_cast<std::ptrdiff_t>(span_offset + length)};
  }

  std::vector<std::byte> _source = std::vector<std::byte>(span_size);
  std::vector<std::byte> _dst0   = std::vector<std::byte>(60);
  std::vector<std::byte> _dst1   = std::vector<std::byte>(30);
  std::vector<std::byte> _dst2   = std::vector<std::byte>(50);
  CallbackContext _ctx;
  std::size_t _fed{0};
};

}  // namespace

TEST_F(RemoteCallbackTest, single_buffer_when_no_segments)
{
  // The easy backend leaves `segments` empty and writes the whole span to one buffer.
  std::vector<std::byte> destination(span_size);
  CallbackContext ctx{destination.data(), span_size};
  EXPECT_EQ(callback_host_memory(reinterpret_cast<char*>(_source.data()), 1, span_size, &ctx),
            span_size);
  EXPECT_EQ(destination, _source);
}

TEST_F(RemoteCallbackTest, scatter_in_one_chunk)
{
  EXPECT_EQ(feed(span_size), span_size);

  EXPECT_EQ(_dst0, expected(0, 60));
  EXPECT_EQ(_dst1, expected(110, 30));
  EXPECT_EQ(_dst2, expected(190, 50));
  EXPECT_EQ(_ctx.offset, static_cast<std::ptrdiff_t>(span_size));
}

TEST_F(RemoteCallbackTest, cursor_survives_many_small_chunks)
{
  // libcurl hands over arbitrary sizes, so a segment can take many calls and a call can cross
  // several segments. 7 does not divide any boundary here.
  for (std::size_t sent = 0; sent < span_size; sent += 7) {
    feed(std::min<std::size_t>(7, span_size - sent));
  }

  EXPECT_EQ(_dst0, expected(0, 60));
  EXPECT_EQ(_dst1, expected(110, 30));
  EXPECT_EQ(_dst2, expected(190, 50));
}

TEST_F(RemoteCallbackTest, chunk_straddles_a_gap)
{
  // Ends mid-segment, then crosses the rest of S0, the whole first gap and into S1.
  feed(50);
  feed(80);

  EXPECT_EQ(_dst0, expected(0, 60));
  EXPECT_THAT(std::vector<std::byte>(_dst1.begin(), _dst1.begin() + 20),
              testing::ElementsAreArray(expected(110, 20)));
  EXPECT_EQ(_ctx.segment_index, 1UL) << "still filling S1";
}

TEST_F(RemoteCallbackTest, gap_bytes_are_never_copied)
{
  feed(span_size);

  // Nothing may be written past a segment's own length.
  EXPECT_EQ(_dst0.size(), 60UL);
  EXPECT_NE(_dst0.back(), _source[60]) << "a gap byte leaked into S0";
  EXPECT_EQ(_dst1.front(), _source[110]);
  EXPECT_EQ(_dst2.front(), _source[190]);
}

TEST_F(RemoteCallbackTest, overflow_is_reported)
{
  EXPECT_EQ(feed(span_size), span_size);
  EXPECT_EQ(callback_host_memory(reinterpret_cast<char*>(_source.data()), 1, 1, &_ctx),
            CURL_WRITEFUNC_ERROR);
  EXPECT_TRUE(_ctx.overflow_error);
}

TEST_F(RemoteCallbackTest, retry_rewinds_the_cursor)
{
  feed(200);
  ASSERT_GT(_ctx.segment_index, 0UL);

  _ctx.reset_for_retry();
  EXPECT_EQ(_ctx.offset, 0);
  EXPECT_EQ(_ctx.segment_index, 0UL);
  EXPECT_FALSE(_ctx.overflow_error);

  // A retry re-sends the whole span, which must land exactly as it would have the first time.
  _fed = 0;
  feed(span_size);
  EXPECT_EQ(_dst0, expected(0, 60));
  EXPECT_EQ(_dst1, expected(110, 30));
  EXPECT_EQ(_dst2, expected(190, 50));
}
