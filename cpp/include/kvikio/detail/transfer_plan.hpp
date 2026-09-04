/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <kvikio/shim/cuda.hpp>

namespace kvikio {
class RemoteHandle;

namespace detail {

// The transfer planner turns N caller requests into M planned transfers. With task_size 60 and
// coalesce_max_gap 10, these four requests become four transfers and five segments:
//
//   requests   ####.####.....###...####################
//              R0   R1       R2    R3
//   transfers  [-------]     [-]   [----------][------]
//              T0            T1    T2          T3
//   segments   [==] [==]     [=]   [==========][======]
//              S0   S1       S2    S3          S4
//
//   merge   R0 + R1   gap of 5 fits in coalesce_max_gap
//   keep    R1, R2    gap of 25 is too wide
//   split   R3        size 100 exceeds task_size
//
// Segments cover only wanted bytes, so T0 needs two of them and the others need one.

/**
 * @brief One read request.
 */
struct TransferPlanRequest {
  RemoteHandle* handle{nullptr};    ///< Non-owning remote file handle.
  CUcontext cuda_context{nullptr};  ///< Null for host buffer.
  void* dst{nullptr};               ///< Start of the destination buffer.
  std::size_t file_offset{0};       ///< Offset of the first requested byte in the remote file.
  std::size_t size{0};              ///< Zero-size requests are excluded from the plan.
};

/**
 * @brief Options controlling how requests become transfers.
 */
struct TransferPlanOptions {
  /**
   * @brief Maximum span of one transfer. Coalescing never exceeds it and a large request is split.
   */
  std::size_t task_size{0};

  /**
   * @brief Merge two ranges separated by at most this many unwanted bytes.
   *
   * Trade bandwidth for round trips. Special values:
   * - 0 merges only adjacent ranges.
   * - `std::nullopt` disables coalescing.
   */
  std::optional<std::size_t> coalesce_max_gap{std::nullopt};
};

/**
 * @brief One contiguous piece of a transfer's span that must be copied into a caller buffer.
 */
struct TransferSegment {
  std::size_t span_offset;    ///< Position in the span.
  std::size_t length;         ///< Number of bytes to copy.
  void* dst;                  ///< Destination buffer.
  std::size_t request_index;  ///< Index into the caller's request span.
};

/**
 * @brief One byte-range request to issue, plus the segments it fills.
 */
struct PlannedTransfer {
  RemoteHandle* handle;       ///< Non-owning remote file handle.
  CUcontext cuda_context;     ///< Null for host buffer.
  std::size_t file_offset;    ///< Where the span starts in the remote file.
  std::size_t size;           ///< Span length, gaps included. Never exceeds `task_size`.
  std::size_t segment_begin;  ///< Starting segment index (inclusive).
  std::size_t segment_end;    ///< Ending segment index (exclusive).
};

/**
 * @brief The transfers to issue for one batch of requests.
 */
struct TransferPlan {
  /**
   * @brief What to fetch. Each entry becomes one HTTP range request.
   *
   * Coalescing merges requests into one entry, whereas splitting spreads one over several.
   */
  std::vector<PlannedTransfer> transfers;

  /**
   * @brief A flat vector indicating where every fetched piece goes, sliced by
   * `PlannedTransfer::segment_begin` and `segment_end`.
   *
   *   transfers  [-------]     [-]   [----------][------]
   *              T0            T1    T2          T3
   *
   *   segments   [==] [==]     [=]   [==========][======]
   *              S0   S1       S2    S3          S4
   *
   * With the example above, T0 owns [0,2), T1 [2,3), T2 [3,4) and T3 [4,5).
   */
  std::vector<TransferSegment> segments;

  /**
   * @brief Number of transfers for each request.
   */
  std::vector<std::size_t> transfers_per_request;

  /**
   * @brief Total gap bytes that will be fetched and discarded.
   */
  std::size_t overread_bytes{0};
};

/**
 * @brief Group, sort, merge and split read requests into the transfers.
 *
 * @param requests The requests to plan. Overlapping and duplicate ranges are not merged.
 * @param opts Planning options.
 * @return The plan. Empty input yields an empty plan.
 * @exception std::invalid_argument if `opts.task_size` is zero.
 */
TransferPlan build_transfer_plan(std::span<TransferPlanRequest const> requests,
                                 TransferPlanOptions const& opts);

}  // namespace detail
}  // namespace kvikio
