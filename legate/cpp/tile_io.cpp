/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "legate_mapping.hpp"
#include "task_opcodes.hpp"

#include <kvikio/file_handle.hpp>

namespace {

/**
 * @brief Get the tile coordinate based on a task index
 *
 * @param task_index Task index
 * @param tile_start The start tile coordinate
 * @return Tile coordinate
 */
legate::DomainPoint get_tile_coord(legate::DomainPoint task_index,
                                   legate::Span<const uint64_t>& tile_start)
{
  for (uint32_t i = 0; i < task_index.dim; ++i) {
    task_index[i] += tile_start[i];
  }
  return task_index;
}

/**
 * @brief Get the file path of a tile
 *
 * @param dirpath The path to the root directory of the Zarr file
 * @param tile_coord The coordinate of the tile
 * @param delimiter The delimiter
 * @return Path to the file representing the requested tile
 */
std::filesystem::path get_file_path(const std::string& dirpath,
                                    const legate::DomainPoint& tile_coord,
                                    const std::string& delimiter = ".")
{
  std::stringstream ss;
  for (int32_t idx = 0; idx < tile_coord.dim; ++idx) {
    if (idx != 0) { ss << delimiter; }
    ss << tile_coord[idx];
  }
  return std::filesystem::path(dirpath) / ss.str();
}

/**
 * @brief Functor for tiling read or write Legate store to or from disk using KvikIO
 *
 * @tparam IsReadOperation Whether the operation is a read or a write operation
 * @param context The Legate task context
 * @param store The Legate store to read or write
 */
template <bool IsReadOperation>
struct tile_read_write_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::TaskContext& context, legate::Store& store)
  {
    using DTYPE                             = legate::legate_type_of<CODE>;
    const auto task_index                   = context.get_task_index();
    const std::string path                  = context.scalars().at(0).value<std::string>();
    legate::Span<const uint64_t> tile_shape = context.scalars().at(1).values<uint64_t>();
    legate::Span<const uint64_t> tile_start = context.scalars().at(2).values<uint64_t>();
    const auto tile_coord                   = get_tile_coord(task_index, tile_start);
    const auto filepath                     = get_file_path(path, tile_coord);

    auto shape        = store.shape<DIM>();
    auto shape_volume = shape.volume();
    if (shape_volume == 0) { return; }
    size_t nbytes = shape_volume * sizeof(DTYPE);

    // We know that the accessor is contiguous because we set `policy.exact = true`
    // in `Mapper::store_mappings()`.
    if constexpr (IsReadOperation) {
      kvikio::FileHandle f(filepath, "r");
      auto* data = store.write_accessor<DTYPE, DIM>().ptr(shape);
      f.pread(data, nbytes).get();
    } else {
      kvikio::FileHandle f(filepath, "w");
      const auto* data = store.read_accessor<DTYPE, DIM>().ptr(shape);
      f.pwrite(data, nbytes).get();
    }
  }
};

/**
 * @brief Flatten the domain point to a 1D point
 *
 * @param lo_dp Lower point
 * @param hi_dp High point
 * @param point_dp The domain point to flatten
 * @return The flatten domain point
 */
template <int32_t DIM>
size_t linearize(const legate::DomainPoint& lo_dp,
                 const legate::DomainPoint& hi_dp,
                 const legate::DomainPoint& point_dp)
{
  const legate::Point<DIM> lo      = lo_dp;
  const legate::Point<DIM> hi      = hi_dp;
  const legate::Point<DIM> point   = point_dp - lo_dp;
  const legate::Point<DIM> extents = hi - lo + legate::Point<DIM>::ONES();
  size_t idx                       = 0;
  for (int32_t dim = 0; dim < DIM; ++dim) {
    idx = idx * extents[dim] + point[dim];
  }
  return idx;
}

/**
 * @brief Functor for tiling read Legate store by offsets from disk using KvikIO
 *
 * @param context The Legate task context
 * @param store The Legate output store
 */
struct tile_read_by_offsets_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::TaskContext& context, legate::Store& store)
  {
    using DTYPE                             = legate::legate_type_of<CODE>;
    const auto task_index                   = context.get_task_index();
    const auto launch_domain                = context.get_launch_domain();
    const std::string path                  = context.scalars().at(0).value<std::string>();
    legate::Span<const uint64_t> offsets    = context.scalars().at(1).values<uint64_t>();
    legate::Span<const uint64_t> tile_shape = context.scalars().at(2).values<uint64_t>();

    // Flatten task index
    uint32_t flatten_task_index =
      linearize<DIM>(launch_domain.lo(), launch_domain.hi(), task_index);

    auto shape        = store.shape<DIM>();
    auto shape_volume = shape.volume();
    if (shape_volume == 0) { return; }
    size_t nbytes = shape_volume * sizeof(DTYPE);
    std::array<size_t, DIM> strides{};

    // We know that the accessor is contiguous because we set `policy.exact = true`
    // in `Mapper::store_mappings()`.
    kvikio::FileHandle f(path, "r");
    auto* data = store.write_accessor<DTYPE, DIM>().ptr(shape, strides.data());
    f.pread(data, nbytes, offsets[flatten_task_index]).get();
  }
};

}  // namespace

namespace legate_kvikio {

/**
 * @brief Write a tiled Legate store to disk using KvikIO
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *     - tile_shape: tuple of int64_t
 *     - tile_start: tuple of int64_t
 *   - inputs:
 *     - buffer: store (any dtype)
 *
 * NB: the store must be contigues. To make Legate in force this,
 *     set `policy.exact = true` in `Mapper::store_mappings()`.
 *
 */
class TileWriteTask : public Task<TileWriteTask, TaskOpCode::OP_TILE_WRITE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    legate::Store& store = context.inputs().at(0);
    legate::double_dispatch(store.dim(), store.code(), tile_read_write_fn<false>{}, context, store);
  }

  static void gpu_variant(legate::TaskContext& context)
  {
    // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the CPU variant.
    cpu_variant(context);
  }
};

/**
 * @brief Read a tiled Legate store to disk using KvikIO
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *     - tile_shape: tuple of int64_t
 *     - tile_start: tuple of int64_t
 *   - outputs:
 *     - buffer: store (any dtype)
 *
 * NB: the store must be contigues. To make Legate in force this,
 *     set `policy.exact = true` in `Mapper::store_mappings()`.
 *
 */
class TileReadTask : public Task<TileReadTask, TaskOpCode::OP_TILE_READ> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    legate::Store& store = context.outputs().at(0);
    legate::double_dispatch(store.dim(), store.code(), tile_read_write_fn<true>{}, context, store);
  }

  static void gpu_variant(legate::TaskContext& context)
  {
    // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the CPU variant.
    cpu_variant(context);
  }
};

/**
 * @brief Read a tiled Legate store by offset to disk using KvikIO
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *     - offsets: tuple of int64_t
 *     - tile_shape: tuple of int64_t
 *   - outputs:
 *     - buffer: store (any dtype)
 *
 * NB: the store must be contigues. To make Legate in force this,
 *     set `policy.exact = true` in `Mapper::store_mappings()`.
 *
 */
class TileReadByOffsetsTask
  : public Task<TileReadByOffsetsTask, TaskOpCode::OP_TILE_READ_BY_OFFSETS> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    legate::Store& store = context.outputs().at(0);
    legate::double_dispatch(store.dim(), store.code(), tile_read_by_offsets_fn{}, context, store);
  }

  static void gpu_variant(legate::TaskContext& context)
  {
    // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the CPU variant.
    cpu_variant(context);
  }
};

}  // namespace legate_kvikio

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate_kvikio::TileWriteTask::register_variants();
  legate_kvikio::TileReadTask::register_variants();
  legate_kvikio::TileReadByOffsetsTask::register_variants();
}

}  // namespace
