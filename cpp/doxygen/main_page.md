# Welcome to KvikIO's C++ documentation!

KvikIO is a Python and C++ library for high performance file IO. It provides C++ and Python
bindings to [cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
which enables [GPUDirect Storage (GDS)](https://developer.nvidia.com/blog/gpudirect-storage/).
KvikIO also works efficiently when GDS isn't available and can read/write both host and device data seamlessly.

KvikIO C++ is part of the [RAPIDS](https://rapids.ai/) suite of open-source software libraries for GPU-accelerated data science.

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

For convenience we release Conda packages that makes it easy to include KvikIO in your CMake projects.

### Conda/Mamba

We strongly recommend using [mamba](https://github.com/mamba-org/mamba) in place of conda, which we will do throughout the documentation.

Install the **stable release** from the ``rapidsai`` channel with the following:

```sh
# Install in existing environment
mamba install -c rapidsai -c conda-forge libkvikio

# Create new environment (CUDA 13)
mamba create -n libkvikio-env -c rapidsai -c conda-forge cuda-version=13.0 libkvikio

# Create new environment (CUDA 12)
mamba create -n libkvikio-env -c rapidsai -c conda-forge cuda-version=12.9 libkvikio
```

Install the **nightly release** from the ``rapidsai-nightly`` channel with the following:

```sh
# Install in existing environment
mamba install -c rapidsai-nightly -c conda-forge libkvikio

# Create new environment (CUDA 13)
mamba create -n libkvikio-env -c rapidsai-nightly -c conda-forge python=3.13 cuda-version=13.0 libkvikio

# Create new environment (CUDA 12)
mamba create -n libkvikio-env -c rapidsai-nightly -c conda-forge python=3.13 cuda-version=12.9 libkvikio
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
When KvikIO is running in compatibility mode, it doesn't load `libcufile.so`. Instead, reads and writes are done using POSIX. Notice, this is not the same as the compatibility mode in cuFile. It is possible that KvikIO performs I/O in the non-compatibility mode by using the cuFile library, but the cuFile library itself is configured to operate in its own compatibility mode. For more details, refer to [cuFile compatibility mode](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-compatibility-mode) and [cuFile environment variables](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#environment-variables)

The environment variable `KVIKIO_COMPAT_MODE` has three options (case-insensitive):
  - `ON` (aliases: `TRUE`, `YES`, `1`): Enable the compatibility mode.
  - `OFF` (aliases: `FALSE`, `NO`, `0`): Disable the compatibility mode, and enforce cuFile I/O. GDS will be activated if the system requirements for cuFile are met and cuFile is properly configured. However, if the system is not suited for cuFile, I/O operations under the `OFF` option may error out.
  - `AUTO`: Try cuFile I/O first, and fall back to POSIX I/O if the system requirements for cuFile are not met.

Under `AUTO`, KvikIO falls back to the compatibility mode:
  - when `libcufile.so` cannot be found.
  - when running in Windows Subsystem for Linux (WSL).
  - when `/run/udev` isn't readable, which typically happens when running inside a docker image not launched with `--volume /run/udev:/run/udev:ro`.

This setting can also be programmatically controlled by `defaults::set_compat_mode()` and `defaults::compat_mode_reset()`.


#### Thread Pool (KVIKIO_NTHREADS)
KvikIO can use multiple threads for IO automatically. Set the environment variable `KVIKIO_NTHREADS` to the number of threads in the thread pool. If not set, the default value is 1.

This setting can also be controlled by `defaults::thread_pool_nthreads()` and `defaults::thread_pool_nthreads_reset()`.

#### Task Size (KVIKIO_TASK_SIZE)
KvikIO splits parallel IO operations into multiple tasks. Set the environment variable `KVIKIO_TASK_SIZE` to the maximum task size (in bytes). If not set, the default value is 4194304 (4 MiB).

This setting can also be controlled by `defaults::task_size()` and `defaults::task_size_reset()`.

#### GDS Threshold (KVIKIO_GDS_THRESHOLD)
To improve performance of small IO requests, `.pread()` and `.pwrite()` implement a shortcut that circumvents the threadpool and uses the POSIX backend directly. Set the environment variable `KVIKIO_GDS_THRESHOLD` to the minimum size (in bytes) to use GDS. If not set, the default value is 1048576 (1 MiB).

This setting can also be controlled by `defaults::gds_threshold()` and `defaults::gds_threshold_reset()`.

#### Size of the Bounce Buffer (KVIKIO_GDS_THRESHOLD)
KvikIO might have to use intermediate host buffers (one per thread) when copying between files and device memory. Set the environment variable `KVIKIO_BOUNCE_BUFFER_SIZE` to the size (in bytes) of these "bounce" buffers. If not set, the default value is 16777216 (16 MiB).

This setting can also be controlled by `defaults::bounce_buffer_size()` and `defaults::bounce_buffer_size_reset()`.

#### HTTP Retries

The behavior when a remote IO read returns a error can be controlled through the `KVIKIO_HTTP_STATUS_CODES` and `KVIKIO_HTTP_MAX_ATTEMPTS` environment variables.
`KVIKIO_HTTP_STATUS_CODES` controls the status codes to retry, and `KVIKIO_HTTP_MAX_ATTEMPTS` controls the maximum number of attempts to make before throwing an exception.

When a response with a status code in the list of retryable codes is received, KvikIO will wait for some period of time before retrying the request.
It will keep retrying until reaching the maximum number of attempts.

By default, KvikIO will retry responses with the following status codes:

- 429
- 500
- 502
- 503
- 504

KvikIO will, by default, make three attempts per read.
Note that if you're reading a large file that has been split into multiple reads through the KvikIO's task size setting, then *each* task will be retried up to the maximum number of attempts.

These settings can also be controlled by `defaults::http_max_attempts()`, `defaults::http_max_attempts_reset()`, `defaults::http_status_codes()`, and `defaults::http_status_codes_reset()`.

#### Remote Verbose (KVIKIO_REMOTE_VERBOSE)
For debugging HTTP requests, you can enable verbose output that shows detailed information about HTTP communication including headers, request/response bodies, connection details, and SSL handshake information.

Set the environment variable `KVIKIO_REMOTE_VERBOSE` to `true`, `on`, `yes`, or `1` (case-insensitive) to enable verbose output. Otherwise, verbose output is disabled by default.

**Warning** this may show sensitive contents from headers and data.

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
