// nvcc test.cu -o test.bin -arch native

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Macro to check CUDA API error code
#define CHECK(err_code) cuda_check(err_code, __FILE__, __LINE__)
inline void cuda_check(cudaError_t err_code, const char* file, int line)
{
  if (err_code != cudaSuccess) {
    int dev{};
    cudaGetDevice(&dev);
    std::stringstream ss;
    ss << "CUDA runtime error: device = " << dev << ", " << cudaGetErrorName(err_code) << "("
       << err_code << "): " << cudaGetErrorString(err_code) << " in " << file << " at line " << line
       << std::endl;
    throw std::runtime_error(ss.str());
  }
}

class TestManager {
 public:
  TestManager()
  {
    CHECK(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreate(&_e_start));
    CHECK(cudaEventCreate(&_e_end));
    CHECK(cudaHostAlloc(&_host_buf, _num_bytes, cudaHostAllocDefault));
    CHECK(cudaMalloc(&_dev_buf, _num_bytes));

    // Initialize host memory
    std::memset(_host_buf, 0, _num_bytes);
  }

  ~TestManager()
  {
    try {
      CHECK(cudaStreamDestroy(_stream));
      CHECK(cudaEventDestroy(_e_start));
      CHECK(cudaEventDestroy(_e_end));
      CHECK(cudaFreeHost(_host_buf));
      CHECK(cudaFree(_dev_buf));
    } catch (std::runtime_error& e) {
      std::cout << e.what();
    }
  }

  void test()
  {
    int num_chunks = _num_bytes / _chunk_bytes;
    auto* host_addr{_host_buf};
    auto* dev_addr{_dev_buf};

    // H2D copy in chunks
    for (int i = 0; i < num_chunks; ++i) {
      std::cout << "    chunk index: " << i << ",";
      float ms{0};

      // Front seat: lower bandwidth
      {
        CHECK(cudaEventRecord(_e_start, _stream));
        CHECK(cudaMemcpyAsync(
          dev_addr, host_addr, _chunk_bytes, cudaMemcpyKind::cudaMemcpyDefault, _stream));
        CHECK(cudaEventRecord(_e_end, _stream));
        CHECK(cudaEventSynchronize(_e_end));
        CHECK(cudaEventElapsedTime(&ms, _e_start, _e_end));
        std::cout << "    bw: " << std::setw(10) << _chunk_bytes / ms / 1024 / 1024 << " [MiB/s], ";
      }

      // Rear seat: higher bandwidth
      {
        CHECK(cudaEventRecord(_e_start, _stream));
        CHECK(cudaMemcpyAsync(
          dev_addr, host_addr, _chunk_bytes, cudaMemcpyKind::cudaMemcpyDefault, _stream));
        CHECK(cudaEventRecord(_e_end, _stream));
        CHECK(cudaEventSynchronize(_e_end));
        CHECK(cudaEventElapsedTime(&ms, _e_start, _e_end));
        std::cout << "    bw: " << std::setw(10) << _chunk_bytes / ms / 1024 / 1024 << " [MiB/s]\n";
      }

      // Update the host and device addresses for the next copy
      host_addr += _chunk_bytes;
      dev_addr += _chunk_bytes;
    }
  }

 private:
  unsigned char* _host_buf{nullptr};
  unsigned char* _dev_buf{nullptr};
  std::size_t _num_bytes{1024u * 1024u * 128u};
  std::size_t _chunk_bytes{1024u * 1024u * 4u};
  cudaStream_t _stream{};
  cudaEvent_t _e_start{};
  cudaEvent_t _e_end{};
};

int main()
{
  TestManager tm;
  tm.test();

  return 0;
}