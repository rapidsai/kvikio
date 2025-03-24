/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <kvikio/threadpool_wrapper.hpp>
#include <optional>
#include "BS_thread_pool.hpp"

namespace kvikio {

namespace this_thread {
template <>
bool is_from_pool<BS_thread_pool>()
{
  return BS::this_thread::get_pool().has_value();
}

template <>
std::optional<std::size_t> index<BS_thread_pool>()
{
  return BS::this_thread::get_index();
}

}  // namespace this_thread

}  // namespace kvikio
