/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <cufile.h>

char const* GetCuErrorString(CUresult cu_result)
{
  char const* description;
  if (cuGetErrorName(cu_result, &description) != CUDA_SUCCESS) description = "unknown cuda error";
  return description;
}

std::string cuFileGetErrorString(int error_code)
{
  return IS_CUFILE_ERR(error_code) ? std::string(CUFILE_ERRSTR(error_code))
                                   : std::string(std::strerror(error_code));
}

std::string cuFileGetErrorString(CUfileError_t status)
{
  std::string error = cuFileGetErrorString(status.err);
  if (IS_CUDA_ERR(status)) { error.append(".").append(GetCuErrorString(status.cu_err)); }
  return error;
}

/** @brief RAII wrapper for a file descriptor and the corresponding cuFile handle. */
class cufile_file {
 public:
  /**
   * @brief Construct a file wrapper.
   *
   * Should not be called directly; use the following factory methods instead.
   *
   * @param file_descriptor A valid file descriptor.
   */
  explicit cufile_file(int file_descriptor) : file_descriptor_{file_descriptor}
  {
    CUfileDescr_t cufile_descriptor{CU_FILE_HANDLE_TYPE_OPAQUE_FD, file_descriptor_};
    auto const status = cuFileHandleRegister(&cufile_handle_, &cufile_descriptor);
    if (status.err != CU_FILE_SUCCESS) {
      close(file_descriptor_);
      throw std::logic_error("Failed to register cuFile handle: " + cuFileGetErrorString(status));
    }
  }

  /**
   * @brief Factory method to create a file wrapper for reading.
   *
   * @param path Absolute path of the file to read from. This file must exist.
   * @return std::unique_ptr<cufile_file> for reading.
   */
  static auto make_reader(char const* path)
  {
    auto const file_descriptor = open(path, O_RDONLY | O_DIRECT);
    if (file_descriptor < 0) {
      throw std::logic_error("Failed to open file to read: " + cuFileGetErrorString(errno));
    }
    return std::make_unique<cufile_file>(file_descriptor);
  }

  /**
   * @brief Factory method to create a file wrapper for writing.
   *
   * @param path Absolute path of the file to write to. This creates the file if it does not already
   * exist.
   * @return std::unique_ptr<cufile_file> for writing.
   */
  static auto make_writer(char const* path)
  {
    auto const file_descriptor = open(path, O_CREAT | O_WRONLY | O_DIRECT, S_IRUSR | S_IWUSR);
    if (file_descriptor < 0) {
      throw std::logic_error("Failed to open file to write: " + cuFileGetErrorString(errno));
    }
    return std::make_unique<cufile_file>(file_descriptor);
  }

  // Disable copy (and move) semantics.
  cufile_file(cufile_file const&)            = delete;
  cufile_file& operator=(cufile_file const&) = delete;

  /** @brief Destroy the file wrapper by de-registering the cuFile handle and closing the file. */
  ~cufile_file() noexcept
  {
    cuFileHandleDeregister(cufile_handle_);
    close(file_descriptor_);
  }

  /**
   * @brief Read the file into a device buffer.
   *
   * @param buffer Device buffer to read the file content into.
   * @param file_offset Starting offset from which to read the file.
   */
  void read(void* buffer,
            std::size_t size,
            std::size_t file_offset,
            std::size_t device_offset) const
  {
    auto const status = cuFileRead(cufile_handle_, buffer, size, file_offset, device_offset);

    if (status < 0) {
      if (IS_CUFILE_ERR(status)) {
        throw std::logic_error("Failed to read file into buffer: " + cuFileGetErrorString(status));
      } else {
        throw std::logic_error("Failed to read file into buffer: " + cuFileGetErrorString(errno));
      }
    }
  }

  void write(void* buffer, std::size_t size, std::size_t file_offset, std::size_t buffer_offset)
  {
    auto const status = cuFileWrite(cufile_handle_, buffer, size, file_offset, buffer_offset);
    if (status < 0) {
      if (IS_CUFILE_ERR(status)) {
        throw std::logic_error("Failed to write file from buffer: " + cuFileGetErrorString(status));
      } else {
        throw std::logic_error("Failed to write file from buffer: " + cuFileGetErrorString(errno));
      }
    }
  }

 private:
  /// The underlying file descriptor.
  int file_descriptor_;
  /// The registered cuFile handle.
  CUfileHandle_t cufile_handle_{};
};

class cufile_driver {
 public:
  cufile_driver()
  {
    auto const status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
      throw std::logic_error("Failed to initialize cuFile driver: " + cuFileGetErrorString(status));
    }
  }

  cufile_driver(cufile_driver const&)            = delete;
  cufile_driver& operator=(cufile_driver const&) = delete;

  ~cufile_driver() { cuFileDriverClose(); }
};

extern "C" {
#include <jni.h>

JNIEXPORT jlong JNICALL Java_ai_rapids_kvikio_cufile_CuFileDriver_create(JNIEnv* env, jclass)
{
  try {
    return reinterpret_cast<jlong>(new cufile_driver());
  } catch (std::exception const& e) {
    jlong default_ret_val = 0;
    if (env->ExceptionOccurred()) { return default_ret_val; }

    jclass exceptionClass = env->FindClass("java/lang/Throwable");
    if (exceptionClass != NULL) { env->ThrowNew(exceptionClass, e.what()); }
    return default_ret_val;
  }
}

JNIEXPORT void JNICALL Java_ai_rapids_kvikio_cufile_CuFileDriver_destroy(JNIEnv* env,
                                                                         jclass,
                                                                         jlong pointer)
{
  delete reinterpret_cast<cufile_driver*>(pointer);
}

JNIEXPORT void JNICALL Java_ai_rapids_kvikio_cufile_CuFileHandle_destroy(JNIEnv* env,
                                                                         jclass,
                                                                         jlong pointer)
{
  delete reinterpret_cast<cufile_file*>(pointer);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_kvikio_cufile_CuFileReadHandle_create(JNIEnv* env,
                                                                             jclass,
                                                                             jstring path)
{
  auto file = cufile_file::make_reader(env->GetStringUTFChars(path, nullptr));
  return reinterpret_cast<jlong>(file.release());
}

JNIEXPORT void JNICALL Java_ai_rapids_kvikio_cufile_CuFileReadHandle_readFile(JNIEnv* env,
                                                                              jclass,
                                                                              jlong file_pointer,
                                                                              jlong device_pointer,
                                                                              jlong size,
                                                                              jlong file_offset,
                                                                              jlong device_offset)
{
  auto* file_ptr = reinterpret_cast<cufile_file*>(file_pointer);
  auto* dev_ptr  = reinterpret_cast<void*>(device_pointer);
  file_ptr->read(dev_ptr, size, file_offset, device_offset);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_kvikio_cufile_CuFileWriteHandle_create(JNIEnv* env,
                                                                              jclass,
                                                                              jstring path)
{
  auto file = cufile_file::make_writer(env->GetStringUTFChars(path, nullptr));
  return reinterpret_cast<jlong>(file.release());
}

JNIEXPORT void JNICALL
Java_ai_rapids_kvikio_cufile_CuFileWriteHandle_writeFile(JNIEnv* env,
                                                         jclass,
                                                         jlong file_pointer,
                                                         jlong device_pointer,
                                                         jlong size,
                                                         jlong file_offset,
                                                         jlong buffer_offset)
{
  auto* file_ptr = reinterpret_cast<cufile_file*>(file_pointer);
  auto* dev_ptr  = reinterpret_cast<void*>(device_pointer);
  file_ptr->write(dev_ptr, size, file_offset, buffer_offset);
}
}
