# Welcome to KvikIO's C++ documentation!

KvikIO is a Python and C++ library for high performance file IO. It provides C++ and Python
bindings to [cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
which enables [GPUDirect Storage (GDS)](https://developer.nvidia.com/blog/gpudirect-storage/).
KvikIO also works efficiently when GDS isn't available and can read/write both host and device data seamlessly.

KvikIO C++ is a header-only library that is part of the [RAPIDS](https://rapids.ai/) suite of open-source software libraries for GPU-accelerated data science.

---
**Notice** this is the documentation for the C++ library. For the Python documentation, see under [kvikio](https://docs.rapids.ai/api/kvikio/nightly/).


---

## Features

* Object Oriented API.
* Exception handling.
* Concurrent reads and writes using an internal thread pool.
* Non-blocking API.
* Handle both host and device IO seamlessly.

## Installation

KvikIO is a header-only library and as such doesn't need installation.
However, for convenience we release Conda packages that makes it easy
to include KvikIO in your CMake projects.

### Conda/Mamba

We strongly recommend using [mamba](https://github.com/mamba-org/mamba) in place of conda, which we will do throughout the documentation.

Install the **stable release** from the ``rapidsai`` channel with the following:
```sh
# Install in existing environment
mamba install -c rapidsai -c conda-forge libkvikio
# Create new environment (CUDA 11.8)
mamba create -n libkvikio-env -c rapidsai -c conda-forge cuda-version=11.8 libkvikio
# Create new environment (CUDA 12.0)
mamba create -n libkvikio-env -c rapidsai -c conda-forge cuda-version=12.0 libkvikio
```

Install the **nightly release** from the ``rapidsai-nightly`` channel with the following:

```sh
# Install in existing environment
mamba install -c rapidsai-nightly -c conda-forge libkvikio
# Create new environment (CUDA 11.8)
mamba create -n libkvikio-env -c rapidsai-nightly -c conda-forge python=3.10 cuda-version=11.8 libkvikio
# Create new environment (CUDA 12.0)
mamba create -n libkvikio-env -c rapidsai-nightly -c conda-forge python=3.10 cuda-version=12.0 libkvikio
```

---
**Notice** if the nightly install doesn't work, set ``channel_priority: flexible`` in your ``.condarc``.

---

### Include KvikIO in a CMake project
An example of how to include KvikIO in an existing CMake project can be found here:  <https://github.com/rapidsai/kvikio/blob/HEAD/cpp/examples/downstream/>.


### Build from source

To build the C++ example run:

```
./build.sh libkvikio
```

Then run the example:

```
./examples/basic_io
```

## Runtime Settings

#### Compatibility Mode (KVIKIO_COMPAT_MODE)
When KvikIO is running in compatibility mode, it doesn't load `libcufile.so`. Instead, reads and writes are done using POSIX. Notice, this is not the same as the compatibility mode in cuFile. That is cuFile can run in compatibility mode while KvikIO is not.

Set the environment variable `KVIKIO_COMPAT_MODE` to enable/disable compatibility mode. By default, compatibility mode is enabled:
  - when `libcufile.so` cannot be found.
  - when running in Windows Subsystem for Linux (WSL).
  - when `/run/udev` isn't readable, which typically happens when running inside a docker image not launched with `--volume /run/udev:/run/udev:ro`.

#### Thread Pool (KVIKIO_NTHREADS)
KvikIO can use multiple threads for IO automatically. Set the environment variable `KVIKIO_NTHREADS` to the number of threads in the thread pool. If not set, the default value is 1.

#### Task Size (KVIKIO_TASK_SIZE)
KvikIO splits parallel IO operations into multiple tasks. Set the environment variable `KVIKIO_TASK_SIZE` to the maximum task size (in bytes). If not set, the default value is 4194304 (4 MiB).

#### GDS Threshold (KVIKIO_GDS_THRESHOLD)
In order to improve performance of small IO, `.pread()` and `.pwrite()` implement a shortcut that circumvent the threadpool and use the POSIX backend directly. Set the environment variable `KVIKIO_GDS_THRESHOLD` to the minimum size (in bytes) to use GDS. If not set, the default value is 1048576 (1 MiB).


## Example

```cpp
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

For a full runnable example see <https://github.com/rapidsai/kvikio/blob/HEAD/cpp/examples/basic_io.cpp>.
