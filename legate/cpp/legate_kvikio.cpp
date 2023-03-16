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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <utility>

#include "legate_kvikio.hpp"
#include "legate_mapping.hpp"

#include <kvikio/file_handle.hpp>

namespace legate_kvikio {

struct elem_size_fn {
  template <legate::LegateTypeCode DTYPE>
  size_t operator()()
  {
    return sizeof(legate::legate_type_of<DTYPE>);
  }
};

size_t sizeof_legate_type_code(legate::LegateTypeCode code)
{
  return legate::type_dispatch(code, elem_size_fn{});
}

class WriteTask : public Task<WriteTask, TaskOpCode::OP_WRITE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    std::string path     = context.scalars()[0].value<std::string>();
    legate::Store& store = context.inputs()[0];
    auto shape           = store.shape<1>();
    auto acc             = store.read_accessor<char, 1>();
    size_t strides[1];
    const char* data = acc.ptr(shape, strides);
    size_t itemsize  = sizeof_legate_type_code(store.code());
    assert(strides[0] == itemsize);  // Must be contiguous
    size_t nbytes = shape.volume() * itemsize;
    size_t offset = shape.lo.x * itemsize;  // Offset in bytes

    // {
    //   std::stringstream ss;
    //   ss << "WriteTask - path: " << path << ", task_idx: " << context.get_task_index()
    //      << ", shape: " << shape << ", offset: " << offset << ", itemsize: " << itemsize
    //      << ", nbytes: " << nbytes << std::endl;
    //   std::cout << ss.str();
    // }

    kvikio::FileHandle f(path, "w");
    f.pwrite(data, nbytes, offset).get();
  }

  static void gpu_variant(legate::TaskContext& context) { cpu_variant(context); }
};

// static file_handle = nullptr;

class ReadTask : public Task<ReadTask, TaskOpCode::OP_READ> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    std::string path     = context.scalars()[0].value<std::string>();
    legate::Store& store = context.outputs()[0];
    auto shape           = store.shape<1>();
    auto acc             = store.write_accessor<char, 1>();
    size_t strides[1];
    char* data      = acc.ptr(shape, strides);
    size_t itemsize = sizeof_legate_type_code(store.code());
    assert(strides[0] == itemsize);  // Must be contiguous
    size_t nbytes = shape.volume() * itemsize;
    size_t offset = shape.lo.x * itemsize;  // Offset in bytes

    // {
    //   std::stringstream ss;
    //   ss << "ReadTask - path: " << path << ", task_idx: " << context.get_task_index()
    //      << ", shape: " << shape << ", offset: " << offset << ", itemsize: " << itemsize
    //      << ", nbytes: " << nbytes << std::endl;
    //   std::cout << ss.str();
    // }

    kvikio::FileHandle f(path, "r");
    f.pread(data, nbytes, offset).get();
  }

  static void gpu_variant(legate::TaskContext& context) { cpu_variant(context); }
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
