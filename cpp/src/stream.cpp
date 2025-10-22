/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/types.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/stream.hpp>

namespace kvikio {

StreamFuture::StreamFuture(
  void* devPtr_base, std::size_t size, off_t file_offset, off_t devPtr_offset, CUstream stream)
  : _devPtr_base{devPtr_base}, _stream{stream}
{
  KVIKIO_NVTX_FUNC_RANGE();
  // Notice, we allocate the arguments using malloc() as specified in the cuFile docs:
  // <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilewriteasync>
  KVIKIO_EXPECT((_val = static_cast<ArgByVal*>(std::malloc(sizeof(ArgByVal)))) != nullptr,
                "Bad memory allocation",
                std::runtime_error);

  *_val = {
    .size = size, .file_offset = file_offset, .devPtr_offset = devPtr_offset, .bytes_done = 0};
}

StreamFuture::StreamFuture(StreamFuture&& o) noexcept
  : _devPtr_base{std::exchange(o._devPtr_base, nullptr)},
    _stream{std::exchange(o._stream, nullptr)},
    _val{std::exchange(o._val, nullptr)},
    _stream_synchronized{o._stream_synchronized}
{
}

StreamFuture& StreamFuture::operator=(StreamFuture&& o) noexcept
{
  _devPtr_base         = std::exchange(o._devPtr_base, nullptr);
  _stream              = std::exchange(o._stream, nullptr);
  _val                 = std::exchange(o._val, nullptr);
  _stream_synchronized = o._stream_synchronized;
  return *this;
}

std::tuple<void*, std::size_t*, off_t*, off_t*, ssize_t*, CUstream> StreamFuture::get_args() const
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_EXPECT(_val != nullptr, "cannot get arguments from an uninitialized StreamFuture");

  return {_devPtr_base,
          &_val->size,
          &_val->file_offset,
          &_val->devPtr_offset,
          &_val->bytes_done,
          _stream};
}

std::size_t StreamFuture::check_bytes_done()
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_EXPECT(_val != nullptr, "cannot check bytes done on an uninitialized StreamFuture");

  if (!_stream_synchronized) {
    _stream_synchronized = true;
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(_stream));
  }

  CUFILE_CHECK_BYTES_DONE(_val->bytes_done);
  // At this point, we know `_val->bytes_done` is a positive value otherwise
  // CUFILE_CHECK_BYTES_DONE() would have raised an exception.
  return static_cast<std::size_t>(_val->bytes_done);
}

StreamFuture::~StreamFuture() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_val != nullptr) {
    try {
      check_bytes_done();
    } catch (kvikio::CUfileException const& e) {
      std::cerr << e.what() << std::endl;
    }
    std::free(_val);
  }
}

}  // namespace kvikio
