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

#include <benchmark/benchmark.h>
#include <utils/utils.hpp>

namespace kvikio::utils {
void explain_default_metrics()
{
  benchmark::AddCustomContext(
    "Time",
    "The average real time (i.e. wall-clock time) of the entire process per benchmark iteration.");
  benchmark::AddCustomContext(
    "CPU",
    "The average CPU time of the main thread per benchmark iteration. The timer is accumulated "
    "only when the main thread is being executed.");
}
}  // namespace kvikio::utils
