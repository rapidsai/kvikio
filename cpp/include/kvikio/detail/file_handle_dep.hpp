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
#pragma once

#include <sys/stat.h>
#include <sys/types.h>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <kvikio/buffer.hpp>
#include <kvikio/cufile/config.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/stream.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {
class FileHandle;

namespace detail {
class FileHandleDependencyBase {
 public:
  virtual ~FileHandleDependencyBase() = default;
  void set_file_handle(FileHandle* file_handle);

  virtual int open_fd(std::string const& file_path,
                      std::string const& flags,
                      bool o_direct,
                      mode_t mode)                                                       = 0;
  virtual void close_fd(int fd)                                                          = 0;
  virtual CUfileError_t cuFile_handle_register(CUfileHandle_t* fh, CUfileDescr_t* descr) = 0;
  virtual void cuFile_handle_deregister(CUfileHandle_t fh)                               = 0;
  virtual bool is_compat_mode_preferred() const noexcept                                 = 0;
  virtual bool is_compat_mode_preferred(CompatMode compat_mode) const noexcept           = 0;
  virtual bool is_stream_api_available() noexcept                                        = 0;
  virtual std::string const& config_path()                                               = 0;

 protected:
  FileHandle* _file_handle;
};

class FileHandleDependencyProduction : public FileHandleDependencyBase {
 public:
  /**
   * @brief Open file using `open(2)`
   *
   * @param flags Open flags given as a string
   * @param o_direct Append O_DIRECT to `flags`
   * @param mode Access modes
   * @return File descriptor
   */
  virtual int open_fd(std::string const& file_path,
                      std::string const& flags,
                      bool o_direct,
                      mode_t mode) override;
  virtual void close_fd(int fd) override;
  virtual CUfileError_t cuFile_handle_register(CUfileHandle_t* fh, CUfileDescr_t* descr) override;
  virtual void cuFile_handle_deregister(CUfileHandle_t fh) override;
  virtual bool is_compat_mode_preferred() const noexcept override;
  virtual bool is_compat_mode_preferred(CompatMode compat_mode) const noexcept override;
  virtual bool is_stream_api_available() noexcept override;
  virtual std::string const& config_path() override;
};
}  // namespace detail
}  // namespace kvikio
