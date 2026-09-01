/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gmock/gmock.h>

#include <kvikio/detail/transfer_plan.hpp>

using kvikio::RemoteHandle;
using kvikio::detail::build_transfer_plan;
using kvikio::detail::TransferPlan;
using kvikio::detail::TransferPlanOptions;
using kvikio::detail::TransferPlanRequest;

namespace {

// Fabricated pointers. The planner compares them and never dereferences them, so these tests
// need neither a server nor a CUDA context.
RemoteHandle* const handle_a = reinterpret_cast<RemoteHandle*>(0x1000);
RemoteHandle* const handle_b = reinterpret_cast<RemoteHandle*>(0x2000);
CUcontext const context_a    = reinterpret_cast<CUcontext>(0x10);
CUcontext const context_b    = reinterpret_cast<CUcontext>(0x20);

/**
 * @brief Check the invariants that must hold for every plan, whatever the input.
 *
 * Coverage is the strongest one. A request's segments must tile its byte range exactly once, in
 * order, at the matching destination address, which catches any slip in the gap arithmetic, the
 * splitting or the per-transfer rebasing.
 */
void expect_plan_invariants(TransferPlan const& plan,
                            std::span<TransferPlanRequest const> requests,
                            TransferPlanOptions const& opts)
{
  ASSERT_EQ(plan.transfers_per_request.size(), requests.size());

  std::vector<std::size_t> covered(requests.size(), 0);
  std::vector<std::size_t> segments_seen(requests.size(), 0);
  std::size_t total_segments = 0;
  std::size_t overread       = 0;

  for (auto const& transfer : plan.transfers) {
    ASSERT_LE(transfer.segment_begin, transfer.segment_end);
    ASSERT_LE(transfer.segment_end, plan.segments.size());
    if (transfer.segment_begin == transfer.segment_end) {
      ADD_FAILURE() << "transfer at " << transfer.file_offset << " fetches bytes nobody wants";
      continue;
    }
    EXPECT_LE(transfer.size, opts.task_size);
    EXPECT_EQ(plan.segments[transfer.segment_begin].span_offset, 0UL)
      << "a span must begin on wanted bytes";

    std::size_t cursor = 0;
    std::size_t wanted = 0;
    for (auto i = transfer.segment_begin; i < transfer.segment_end; ++i) {
      auto const& segment = plan.segments[i];
      ASSERT_LT(segment.request_index, requests.size());
      auto const& request = requests[segment.request_index];

      EXPECT_GE(segment.span_offset, cursor) << "segments must be sorted and must not overlap";
      EXPECT_LE(segment.span_offset + segment.length, transfer.size)
        << "segment runs past the end of its span";
      cursor = segment.span_offset + segment.length;

      auto const done = covered[segment.request_index];
      EXPECT_EQ(transfer.file_offset + segment.span_offset, request.file_offset + done)
        << "request " << segment.request_index << " is covered out of order or with a hole";
      EXPECT_EQ(segment.dst, static_cast<std::byte*>(request.dst) + done);
      EXPECT_EQ(transfer.handle, request.handle);
      EXPECT_EQ(transfer.cuda_context, request.cuda_context);

      covered[segment.request_index] += segment.length;
      ++segments_seen[segment.request_index];
      wanted += segment.length;
    }

    EXPECT_EQ(cursor, transfer.size) << "a span must end on wanted bytes";
    overread += transfer.size - wanted;
    total_segments += transfer.segment_end - transfer.segment_begin;
  }

  EXPECT_EQ(total_segments, plan.segments.size())
    << "every segment belongs to exactly one transfer";

  // Transfers of one group are contiguous and never go backwards, so a caller can route one
  // reactor per group with a single scan.
  std::vector<std::pair<RemoteHandle*, CUcontext>> group_order;
  for (std::size_t i = 0; i < plan.transfers.size(); ++i) {
    auto const& transfer = plan.transfers[i];
    std::pair const key{transfer.handle, transfer.cuda_context};
    if (!group_order.empty() && group_order.back() == key) {
      EXPECT_GE(transfer.file_offset, plan.transfers[i - 1].file_offset)
        << "transfer " << i << " goes backwards within its group";
      continue;
    }
    EXPECT_EQ(std::find(group_order.begin(), group_order.end(), key), group_order.end())
      << "group of transfer " << i << " is not contiguous";
    group_order.push_back(key);
  }
  EXPECT_EQ(overread, plan.overread_bytes);

  for (std::size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(covered[i], requests[i].size) << "request " << i << " is not covered exactly once";
    EXPECT_EQ(segments_seen[i], plan.transfers_per_request[i]) << "request " << i;
  }
}

class TransferPlanTest : public ::testing::Test {
 protected:
  // Real memory, so the expected `dst` arithmetic is real. The stride keeps buffers disjoint.
  static constexpr std::size_t dst_stride = 1UL << 14;

  std::vector<std::byte> _destinations = std::vector<std::byte>(64 * dst_stride);
  std::vector<TransferPlanRequest> _requests;

  std::size_t add_request(std::size_t file_offset,
                          std::size_t size,
                          RemoteHandle* handle   = handle_a,
                          CUcontext cuda_context = nullptr)
  {
    auto const index = _requests.size();
    _requests.push_back({.handle       = handle,
                         .cuda_context = cuda_context,
                         .dst          = _destinations.data() + index * dst_stride,
                         .file_offset  = file_offset,
                         .size         = size});
    return index;
  }

  [[nodiscard]] TransferPlan build_plan(TransferPlanOptions const& opts)
  {
    auto plan = build_transfer_plan(_requests, opts);
    expect_plan_invariants(plan, _requests, opts);
    return plan;
  }

  [[nodiscard]] void* dst_of(std::size_t request_index) const
  {
    return _requests[request_index].dst;
  }
};

}  // namespace

TEST_F(TransferPlanTest, empty_input)
{
  auto const plan = build_plan({.task_size = 1024});
  EXPECT_TRUE(plan.transfers.empty());
  EXPECT_TRUE(plan.segments.empty());
  EXPECT_TRUE(plan.transfers_per_request.empty());
  EXPECT_EQ(plan.overread_bytes, 0UL);
}

TEST_F(TransferPlanTest, task_size_must_be_positive)
{
  add_request(0, 100);
  EXPECT_THROW(build_transfer_plan(_requests, {.task_size = 0}), std::invalid_argument);
}

TEST_F(TransferPlanTest, single_request)
{
  add_request(4096, 100);
  auto const plan = build_plan({.task_size = 1024});

  ASSERT_EQ(plan.transfers.size(), 1UL);
  EXPECT_EQ(plan.transfers[0].handle, handle_a);
  EXPECT_EQ(plan.transfers[0].file_offset, 4096UL);
  EXPECT_EQ(plan.transfers[0].size, 100UL);
  ASSERT_EQ(plan.segments.size(), 1UL);
  EXPECT_EQ(plan.segments[0].span_offset, 0UL);
  EXPECT_EQ(plan.segments[0].length, 100UL);
  EXPECT_EQ(plan.segments[0].dst, dst_of(0));
  EXPECT_THAT(plan.transfers_per_request, testing::ElementsAre(1UL));
  EXPECT_EQ(plan.overread_bytes, 0UL);
}

TEST_F(TransferPlanTest, request_larger_than_task_size_is_split)
{
  add_request(0, 2500);
  auto const plan = build_plan({.task_size = 1000});

  ASSERT_EQ(plan.transfers.size(), 3UL);
  EXPECT_EQ(plan.transfers[0].file_offset, 0UL);
  EXPECT_EQ(plan.transfers[0].size, 1000UL);
  EXPECT_EQ(plan.transfers[1].file_offset, 1000UL);
  EXPECT_EQ(plan.transfers[1].size, 1000UL);
  EXPECT_EQ(plan.transfers[2].file_offset, 2000UL);
  EXPECT_EQ(plan.transfers[2].size, 500UL);

  // Each piece restarts at span offset 0 and continues into the same destination buffer.
  ASSERT_EQ(plan.segments.size(), 3UL);
  EXPECT_EQ(plan.segments[1].span_offset, 0UL);
  EXPECT_EQ(plan.segments[1].dst, static_cast<std::byte*>(dst_of(0)) + 1000);
  EXPECT_EQ(plan.segments[2].dst, static_cast<std::byte*>(dst_of(0)) + 2000);
  EXPECT_THAT(plan.transfers_per_request, testing::ElementsAre(3UL));
  EXPECT_EQ(plan.overread_bytes, 0UL);
}

TEST_F(TransferPlanTest, huge_task_size_does_not_split)
{
  add_request(4096, 2500);
  auto const plan = build_plan({.task_size = std::numeric_limits<std::size_t>::max()});

  ASSERT_EQ(plan.transfers.size(), 1UL);
  EXPECT_EQ(plan.transfers[0].file_offset, 4096UL);
  EXPECT_EQ(plan.transfers[0].size, 2500UL);
}

TEST_F(TransferPlanTest, adjacent_ranges_merge)
{
  add_request(0, 100);
  add_request(100, 50);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 0});

  ASSERT_EQ(plan.transfers.size(), 1UL);
  EXPECT_EQ(plan.transfers[0].file_offset, 0UL);
  EXPECT_EQ(plan.transfers[0].size, 150UL);
  ASSERT_EQ(plan.segments.size(), 2UL);
  EXPECT_EQ(plan.segments[1].span_offset, 100UL);
  EXPECT_EQ(plan.segments[1].dst, dst_of(1));
  EXPECT_THAT(plan.transfers_per_request, testing::ElementsAre(1UL, 1UL));
  EXPECT_EQ(plan.overread_bytes, 0UL);
}

TEST_F(TransferPlanTest, zero_gap_merges_only_adjacent_ranges)
{
  add_request(0, 100);
  add_request(101, 50);  // One unwanted byte in between.
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 0});

  EXPECT_EQ(plan.transfers.size(), 2UL);
  EXPECT_EQ(plan.overread_bytes, 0UL);
}

TEST_F(TransferPlanTest, gap_within_limit_merges)
{
  add_request(0, 100);
  add_request(150, 50);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 50});

  ASSERT_EQ(plan.transfers.size(), 1UL);
  EXPECT_EQ(plan.transfers[0].size, 200UL);
  ASSERT_EQ(plan.segments.size(), 2UL);
  EXPECT_EQ(plan.segments[0].length, 100UL);
  EXPECT_EQ(plan.segments[1].span_offset, 150UL) << "the gap is skipped, not represented";
  EXPECT_EQ(plan.overread_bytes, 50UL);
}

TEST_F(TransferPlanTest, gap_beyond_limit_does_not_merge)
{
  add_request(0, 100);
  add_request(151, 50);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 50});

  EXPECT_EQ(plan.transfers.size(), 2UL);
  EXPECT_EQ(plan.overread_bytes, 0UL);
}

TEST_F(TransferPlanTest, coalescing_is_off_by_default)
{
  add_request(0, 100);
  add_request(100, 100);
  auto const plan = build_plan({.task_size = 1024});

  ASSERT_EQ(plan.transfers.size(), 2UL) << "adjacent ranges must stay apart without a gap limit";
  EXPECT_EQ(plan.transfers[0].file_offset, 0UL);
  EXPECT_EQ(plan.transfers[1].file_offset, 100UL);
  EXPECT_THAT(plan.transfers_per_request, testing::ElementsAre(1UL, 1UL));
}

TEST_F(TransferPlanTest, merging_stops_at_task_size)
{
  add_request(0, 400);
  add_request(400, 400);
  add_request(800, 400);
  auto const plan = build_plan({.task_size = 1000, .coalesce_max_gap = 0});

  // The first two fill 800 of the 1000-byte cap, and the third would overshoot it.
  ASSERT_EQ(plan.transfers.size(), 2UL);
  EXPECT_EQ(plan.transfers[0].size, 800UL);
  EXPECT_EQ(plan.transfers[1].file_offset, 800UL);
  EXPECT_EQ(plan.transfers[1].size, 400UL);
  EXPECT_THAT(plan.transfers_per_request, testing::ElementsAre(1UL, 1UL, 1UL));
}

TEST_F(TransferPlanTest, merging_up_to_exactly_task_size_is_allowed)
{
  add_request(0, 100);
  add_request(100, 100);
  auto const plan = build_plan({.task_size = 200, .coalesce_max_gap = 0});

  ASSERT_EQ(plan.transfers.size(), 1UL) << "a span may reach `task_size`, it just may not pass it";
  EXPECT_EQ(plan.transfers[0].size, 200UL);
}

TEST_F(TransferPlanTest, overlapping_ranges_do_not_merge)
{
  add_request(0, 100);
  add_request(50, 100);

  EXPECT_EQ(build_plan({.task_size = 1024, .coalesce_max_gap = 1024}).transfers.size(), 2UL);

  // Only a limit this large exercises the overlap check. Below it the gap comparison happens to
  // reject overlaps by unsigned wraparound.
  auto const no_limit = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(build_plan({.task_size = 1024, .coalesce_max_gap = no_limit}).transfers.size(), 2UL);
}

TEST_F(TransferPlanTest, duplicate_ranges_do_not_merge)
{
  add_request(64, 100);
  add_request(64, 100);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 1024});

  ASSERT_EQ(plan.transfers.size(), 2UL);
  EXPECT_EQ(plan.transfers[0].file_offset, 64UL);
  EXPECT_EQ(plan.transfers[1].file_offset, 64UL);
  EXPECT_EQ(plan.segments[0].dst, dst_of(0));
  EXPECT_EQ(plan.segments[1].dst, dst_of(1));
}

TEST_F(TransferPlanTest, unsorted_input_plans_like_sorted_input)
{
  add_request(300, 100);
  add_request(0, 100);
  add_request(200, 100);
  add_request(100, 100);
  TransferPlanOptions const opts{.task_size = 1024, .coalesce_max_gap = 0};
  auto const plan = build_plan(opts);

  ASSERT_EQ(plan.transfers.size(), 1UL) << "the four ranges are adjacent once sorted";
  EXPECT_EQ(plan.transfers[0].file_offset, 0UL);
  EXPECT_EQ(plan.transfers[0].size, 400UL);

  // Segments follow file order, while `request_index` still points back at the caller's ordering.
  ASSERT_EQ(plan.segments.size(), 4UL);
  EXPECT_EQ(plan.segments[0].request_index, 1UL);
  EXPECT_EQ(plan.segments[1].request_index, 3UL);
  EXPECT_EQ(plan.segments[2].request_index, 2UL);
  EXPECT_EQ(plan.segments[3].request_index, 0UL);

  // The input is untouched, which is what keeps the caller's indices meaningful.
  EXPECT_EQ(_requests[0].file_offset, 300UL);
  EXPECT_EQ(_requests[1].file_offset, 0UL);
}

TEST_F(TransferPlanTest, requests_of_different_handles_do_not_merge)
{
  add_request(0, 100, handle_a);
  add_request(100, 100, handle_b);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 1024});

  ASSERT_EQ(plan.transfers.size(), 2UL);
  EXPECT_EQ(plan.transfers[0].handle, handle_a);
  EXPECT_EQ(plan.transfers[1].handle, handle_b);
}

TEST_F(TransferPlanTest, host_and_device_destinations_do_not_merge)
{
  add_request(0, 100, handle_a, nullptr);
  add_request(100, 100, handle_a, context_a);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 1024});

  ASSERT_EQ(plan.transfers.size(), 2UL);
  EXPECT_EQ(plan.transfers[0].cuda_context, nullptr);
  EXPECT_EQ(plan.transfers[1].cuda_context, context_a);
}

TEST_F(TransferPlanTest, different_cuda_contexts_do_not_merge)
{
  add_request(0, 100, handle_a, context_a);
  add_request(100, 100, handle_a, context_b);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 1024});

  ASSERT_EQ(plan.transfers.size(), 2UL);
  EXPECT_EQ(plan.transfers[0].cuda_context, context_a);
  EXPECT_EQ(plan.transfers[1].cuda_context, context_b);
}

TEST_F(TransferPlanTest, groups_keep_first_appearance_order)
{
  add_request(0, 100, handle_b);
  add_request(0, 100, handle_a);
  add_request(100, 100, handle_b);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 0});

  ASSERT_EQ(plan.transfers.size(), 2UL);
  EXPECT_EQ(plan.transfers[0].handle, handle_b) << "handle_b is seen first";
  EXPECT_EQ(plan.transfers[0].size, 200UL);
  EXPECT_EQ(plan.transfers[1].handle, handle_a);
}

TEST_F(TransferPlanTest, zero_size_requests_are_excluded)
{
  add_request(0, 100);
  add_request(100, 0);
  add_request(100, 100);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 0});

  ASSERT_EQ(plan.transfers.size(), 1UL);
  EXPECT_EQ(plan.transfers[0].size, 200UL);
  ASSERT_EQ(plan.segments.size(), 2UL);
  EXPECT_EQ(plan.segments[1].request_index, 2UL);
  EXPECT_THAT(plan.transfers_per_request, testing::ElementsAre(1UL, 0UL, 1UL));
}

TEST_F(TransferPlanTest, only_zero_size_requests)
{
  add_request(0, 0);
  add_request(100, 0);
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 0});

  EXPECT_TRUE(plan.transfers.empty());
  EXPECT_THAT(plan.transfers_per_request, testing::ElementsAre(0UL, 0UL));
}

TEST_F(TransferPlanTest, overread_counts_every_gap)
{
  add_request(0, 10);
  add_request(30, 10);   // 20 gap bytes.
  add_request(45, 10);   // 5 gap bytes.
  add_request(500, 10);  // Beyond the limit, so it starts a second transfer.
  auto const plan = build_plan({.task_size = 1024, .coalesce_max_gap = 20});

  ASSERT_EQ(plan.transfers.size(), 2UL);
  EXPECT_EQ(plan.transfers[0].file_offset, 0UL);
  EXPECT_EQ(plan.transfers[0].size, 55UL);
  EXPECT_EQ(plan.transfers[1].file_offset, 500UL);
  EXPECT_EQ(plan.overread_bytes, 25UL);
}

TEST_F(TransferPlanTest, many_small_ranges_over_two_handles)
{
  // 64-byte ranges with 8-byte holes, alternating between two handles, added back to front so
  // the sort has real work to do. Per handle the stride is 144, so four ranges span 496 and a
  // fifth would pass 512.
  constexpr std::size_t num_ranges = 32;
  for (std::size_t i = num_ranges; i-- > 0;) {
    add_request(i * 72, 64, (i % 2 == 0) ? handle_a : handle_b);
  }
  auto const plan = build_plan({.task_size = 512, .coalesce_max_gap = 128});

  EXPECT_EQ(plan.transfers.size(), 8UL);
  EXPECT_EQ(plan.segments.size(), num_ranges);
  for (auto const& transfer : plan.transfers) {
    EXPECT_LE(transfer.size, 512UL);
  }
  EXPECT_THAT(plan.transfers_per_request, testing::Each(1UL));
}
