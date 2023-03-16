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
#include <iostream>
#include <sstream>
#include <utility>

#include "legate_mapping.hpp"
#include "task_opcodes.hpp"

#include <kvikio/file_handle.hpp>

namespace legate_kvikio {

/**
 * @brief Functor converting Legate type code to size
 */
struct elem_size_fn {
  template <legate::LegateTypeCode DTYPE>
  size_t operator()()
  {
    return sizeof(legate::legate_type_of<DTYPE>);
  }
};

/**
 * @brief Get the size of a Legate type code
 *
 * @param code Legate type code
 * @return The number of bytes
 */
size_t sizeof_legate_type_code(legate::LegateTypeCode code)
{
  return legate::type_dispatch(code, elem_size_fn{});
}

/**
 * @brief Get store argument from task context
 *
 * @tparam IsOutputArgument Whether it is an output or an input argument
 * @param context Legate task context.
 * @param i The argument index
 * @return The i'th argument store argument
 */
template <bool IsOutputArgument>
legate::Store& get_store_arg(legate::TaskContext& context, int i)
{
  if constexpr (IsOutputArgument) { return context.outputs()[i]; }
  return context.inputs()[i];
}

/**
 * @brief Read or write Legate store to or from disk using KvikIO
 *
 * @tparam IsReadOperation Whether the operation is a read or a write operation
 * @param context Legate task context.
 */
template <bool IsReadOperation>
void read_write_store(legate::TaskContext& context)
{
  std::string path     = context.scalars()[0].value<std::string>();
  legate::Store& store = get_store_arg<IsReadOperation>(context, 0);
  auto shape           = store.shape<1>();
  size_t itemsize      = sizeof_legate_type_code(store.code());
  size_t nbytes        = shape.volume() * itemsize;
  size_t offset        = shape.lo.x * itemsize;  // Offset in bytes
  std::array<size_t, 1> strides{};

  // We know that the accessor is contiguous because we set `policy.exact = true`
  // in `Mapper::store_mappings()`.
  // TODO: support of non-contigues stores
  if constexpr (IsReadOperation) {
    kvikio::FileHandle f(path, "r");
    auto* data = store.write_accessor<char, 1>().ptr(shape, strides.data());
    assert(strides[0] == itemsize);
    f.pread(data, nbytes, offset).get();
  } else {
    kvikio::FileHandle f(path, "w");
    const auto* data = store.read_accessor<char, 1>().ptr(shape, strides.data());
    assert(strides[0] == itemsize);
    f.pwrite(data, nbytes, offset).get();
  }
}

/**
 * @brief Write a Legate store to disk using KvikIO
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *   - inputs:
 *     - buffer: 1d store (any dtype)
 */
class WriteTask : public Task<WriteTask, TaskOpCode::OP_WRITE> {
 public:
  static void cpu_variant(legate::TaskContext& context) { read_write_store<false>(context); }

  static void gpu_variant(legate::TaskContext& context) { read_write_store<false>(context); }
};

/**
 * @brief Read a Legate store from disk using KvikIO
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *   - outputs:
 *     - buffer: 1d store (any dtype)
 */
class ReadTask : public Task<ReadTask, TaskOpCode::OP_READ> {
 public:
  static void cpu_variant(legate::TaskContext& context) { read_write_store<true>(context); }

  static void gpu_variant(legate::TaskContext& context) { read_write_store<true>(context); }
};

}  // namespace legate_kvikio

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  legate_kvikio::WriteTask::register_variants();
  legate_kvikio::ReadTask::register_variants();
}

}  // namespace
