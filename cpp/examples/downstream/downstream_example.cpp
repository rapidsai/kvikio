/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <iostream>

#include <kvikio/buffer.hpp>
#include <kvikio/config.hpp>
#include <kvikio/driver.hpp>
#include <kvikio/file_handle.hpp>

using namespace std;

int main()
{
  cout << "KvikIO config: " << endl;
  if (kvikio::config::get_global_compat_mode()) {
    cout << "  Compatibility mode: enabled" << endl;
  } else {
    kvikio::DriverInitializer manual_init_driver;
    cout << "  Compatibility mode: disabled" << endl;
    kvikio::DriverProperties props;
    cout << "DriverProperties: " << endl;
    cout << "  Version: " << props.get_nvfs_major_version() << "." << props.get_nvfs_minor_version()
         << endl;
    cout << "  Allow compatibility mode: " << std::boolalpha << props.get_nvfs_allow_compat_mode()
         << endl;
    cout << "  Pool mode - enabled: " << std::boolalpha << props.get_nvfs_poll_mode()
         << ", threshold: " << props.get_nvfs_poll_thresh_size() << " kb" << endl;
    cout << "  Max pinned memory: " << props.get_max_pinned_memory_size() << " kb" << endl;
  }
}
