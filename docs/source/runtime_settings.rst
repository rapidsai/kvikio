Runtime Settings
================

Compatibility Mode ``KVIKIO_COMPAT_MODE``
-----------------------------------------
When KvikIO is running in compatibility mode, it doesn't load ``libcufile.so``. Instead, reads and writes are done using POSIX. Notice, this is not the same as the compatibility mode in cuFile. It is possible that KvikIO performs I/O in the non-compatibility mode by using the cuFile library, but the cuFile library itself is configured to operate in its own compatibility mode. For more details, refer to `cuFile compatibility mode <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-compatibility-mode>`_ and `cuFile environment variables <https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#environment-variables>`_ .

The environment variable ``KVIKIO_COMPAT_MODE`` has three options:

  * ``ON``: Enable the compatibility mode.
  * ``OFF``: Disable the compatibility mode, and enforce cuFile I/O. GDS will be activated if the system requirements for cuFile are met and cuFile is properly configured. However, if the system is not suited for cuFile, I/O operations under the ``OFF`` option may error out, crash or hang.
  * ``AUTO`` (default): Try cuFile I/O first, and fall back to POSIX I/O if the system requirements for cuFile are not met.

Under ``AUTO``, KvikIO falls back to the compatibility mode:

  * when ``libcufile.so`` cannot be found.
  * when running in Windows Subsystem for Linux (WSL).
  * when ``/run/udev`` isn't readable, which typically happens when running inside a docker image not launched with ``--volume /run/udev:/run/udev:ro``.

This setting can also be programmatically controlled by :py:func:`kvikio.defaults.set_compat_mode` and :py:func:`kvikio.defaults.compat_mode_reset`.

Thread Pool ``KVIKIO_NTHREADS``
-------------------------------
KvikIO can use multiple threads for IO automatically. Set the environment variable ``KVIKIO_NTHREADS`` to the number of threads in the thread pool. If not set, the default value is 1.

This setting can also be controlled by :py:func:`kvikio.defaults.get_num_threads`, :py:func:`kvikio.defaults.num_threads_reset`, and :py:func:`kvikio.defaults.set_num_threads`.

Task Size ``KVIKIO_TASK_SIZE``
------------------------------
KvikIO splits parallel IO operations into multiple tasks. Set the environment variable ``KVIKIO_TASK_SIZE`` to the maximum task size (in bytes). If not set, the default value is 4194304 (4 MiB).

This setting can also be controlled by :py:func:`kvikio.defaults.task_size`, :py:func:`kvikio.defaults.task_size_reset`, and :py:func:`kvikio.defaults.set_task_size`.

GDS Threshold ``KVIKIO_GDS_THRESHOLD``
--------------------------------------
In order to improve performance of small IO, ``.pread()`` and ``.pwrite()`` implement a shortcut that circumvent the threadpool and use the POSIX backend directly. Set the environment variable ``KVIKIO_GDS_THRESHOLD`` to the minimum size (in bytes) to use GDS. If not set, the default value is 1048576 (1 MiB).

This setting can also be controlled by :py:func:`kvikio.defaults.gds_threshold`, :py:func:`kvikio.defaults.gds_threshold_reset`, and :py:func:`kvikio.defaults.set_gds_threshold`.

Size of the Bounce Buffer ``KVIKIO_BOUNCE_BUFFER_SIZE``
-------------------------------------------------------
KvikIO might have to use intermediate host buffers (one per thread) when copying between files and device memory. Set the environment variable ``KVIKIO_BOUNCE_BUFFER_SIZE`` to the size (in bytes) of these "bounce" buffers. If not set, the default value is 16777216 (16 MiB).

This setting can also be controlled by :py:func:`kvikio.defaults.bounce_buffer_size`, :py:func:`kvikio.defaults.bounce_buffer_size_reset`, and :py:func:`kvikio.defaults.set_bounce_buffer_size`.
