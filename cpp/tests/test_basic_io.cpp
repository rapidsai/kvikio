/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <kvikio/file_handle.hpp>

#include "utils.hpp"

using namespace kvikio::test;

TEST(BasicIO, write_read)
{
  TempDir tmp_dir{false};
  auto filepath = tmp_dir.path() / "test";

  auto dev_a = DevBuffer::arange(100);
  auto dev_b = DevBuffer::zero_like(dev_a);

  {
    kvikio::FileHandle f(filepath, "w");
    auto nbytes = f.write(dev_a.ptr, dev_a.nbytes, 0, 0);
    EXPECT_EQ(nbytes, dev_a.nbytes);
  }

  {
    kvikio::FileHandle f(filepath, "r");
    auto nbytes = f.read(dev_b.ptr, dev_b.nbytes, 0, 0);
    EXPECT_EQ(nbytes, dev_b.nbytes);
    expect_equal(dev_a, dev_b);
  }
}
