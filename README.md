# C++ and Python bindings to cuFile

## Summary

This provides C++ and Python bindings to cuFile, which enables GPUDirect Storage (or GDS).

### Features

* Object Oriented API
* Exception handling
* Zarr reader

## Requirements

To install users should have a working Linux machine with CUDA Toolkit
installed (v11.4+) and a working compiler toolchain (C++17 and cmake).

### C++

The C++ bindings are header-only and depends on CUDA Driver and Runtime API.
In order to build and run the example code, CMake is required.

### Python

The Python packages depends on the following packages:

* Cython
* Pip
* Setuptools

For testing:
* pytest
* cupy

## Install

### C++
To build the C++ example, go to the `cpp` subdiretory and run:
```
mkdir build
cd build
cmake ..
make
```
Then run the example:
```
./examples/basic_io
```

### Python

To build and install the extension, go to the `python` subdiretory and run:
```
python -m pip install .
```
One might have to define `CUDA_HOME` to the path to the CUDA installation.

In order to test the installation, run the following:
```
pytest tests/
```

And to test performance, run the following:
```
python benchmarks/single-node-io.py
```


## Examples

### C++
```c++
#include <cstddef>
#include <cuda_runtime.h>
#include <kvikio/file_handle.hpp>
using namespace std;

int main()
{
  // Create two arrays `a` and `b`
  constexpr std::size_t size = 100;
  void *a = nullptr;
  void *b = nullptr;
  cudaMalloc(&a, size);
  cudaMalloc(&b, size);

  // Write `a` to file
  kvikio::FileHandle fw("test-file", "w");
  size_t written = fw.write(a, size);
  fw.close();

  // Read file into `b`
  kvikio::FileHandle fr("test-file", "r");
  size_t read = fr.read(b, size);
  fr.close();

  // Read file into `b` in parallel using 16 threads
  kvikio::default_thread_pool::reset(16);
  {
    kvikio::FileHandle f("test-file", "r");
    future<size_t> future = f.pread(b_dev, sizeof(a), 0);  // Non-blocking
    size_t read = future.get(); // Blocking
    // Notice, `f` closes automatically on destruction.
  }
}
```

### Python
```python
import cupy
import kvikio

a = cupy.arange(100)
f = kvikio.CuFile("test-file", "w")
# Write whole array to file
f.write(a)
f.close()

b = cupy.empty_like(a)
f = kvikio.CuFile("test-file", "r")
# Read whole array from file
f.read(b)
assert all(a == b)
```
