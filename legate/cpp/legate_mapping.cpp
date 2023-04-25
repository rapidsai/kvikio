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

#include <vector>

#include "core/mapping/mapping.h"
#include "legate_mapping.hpp"
#include "task_opcodes.hpp"

namespace legate_kvikio {

class Mapper : public legate::mapping::Mapper {
 public:
  Mapper() {}

  Mapper(const Mapper& rhs)            = delete;
  Mapper& operator=(const Mapper& rhs) = delete;

  // Legate mapping functions

  void set_machine(const legate::mapping::MachineQueryInterface* machine) override
  {
    machine_ = machine;
  }

  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override
  {
    return *options.begin();  // Choose first priority
  }

  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    using legate::mapping::StoreMapping;
    std::vector<StoreMapping> mappings;
    const auto& inputs  = task.inputs();
    const auto& outputs = task.outputs();
    for (const auto& input : inputs) {
      mappings.push_back(StoreMapping::default_mapping(input, options.front()));
      mappings.back().policy.exact = true;
    }
    for (const auto& output : outputs) {
      mappings.push_back(StoreMapping::default_mapping(output, options.front()));
      mappings.back().policy.exact = true;
    }
    return std::move(mappings);
  }

  legate::Scalar tunable_value(legate::TunableID tunable_id) override { return 0; }

 private:
  const legate::mapping::MachineQueryInterface* machine_;
};

static const char* const library_name = "legate_kvikio";

Legion::Logger log_legate_kvikio(library_name);

/*static*/ legate::TaskRegistrar& Registry::get_registrar()
{
  static legate::TaskRegistrar registrar;
  return registrar;
}

void registration_callback()
{
  legate::ResourceConfig config;
  config.max_tasks = OP_NUM_TASK_IDS;

  auto context = legate::Runtime::get_runtime()->create_library(
    library_name, config, std::make_unique<Mapper>());
  Registry::get_registrar().register_all_tasks(context);
}

}  // namespace legate_kvikio

extern "C" {

void legate_kvikio_perform_registration(void)
{
  // Tell the runtime about our registration callback so we hook it
  // in before the runtime starts and make it global so that we know
  // that this call back is invoked everywhere across all nodes
  legate::Core::perform_registration<legate_kvikio::registration_callback>();
}
}
