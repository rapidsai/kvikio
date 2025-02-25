# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


from typing import Any

import kvikio._lib.defaults


class ConfigContextManager:
    def __init__(self, config: dict):
        self._old_properties = {}

        for key, value in config.items():
            self._old_properties[key] = self._get_property(key)
            self._set_property(key, value)

    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        for key, value in self._old_properties.items():
            self._set_property(key, value)

    def _get_property(self, property: str) -> Any:
        if property == "num_threads":
            property = "thread_pool_nthreads"
        func = getattr(kvikio._lib.defaults, property)
        return func()

    def _set_property(self, property: str, value: Any):
        if property == "num_threads":
            property = "thread_pool_nthreads"
        func = getattr(kvikio._lib.defaults, f"set_{property}")
        func(value)


def set(*config: Any) -> ConfigContextManager:
    """Set KvikIO configurations.

    Examples:

    - To set one or more properties

      .. code-block:: python

         kvikio.defaults.set({"prop1": value1, "prop2": value2})

    - To set a single property

      .. code-block:: python

         kvikio.defaults.set("prop", value)

    Parameters
    ----------
    Any
        The configurations. Can either be a single parameter (dict) consisting of one
        or more properties, or two parameters key (string) and value (Any)
        indicating a single property.
    """

    err_msg = (
        "Valid arguments are kvikio.defaults.set(config: dict) or "
        "kvikio.defaults.set(key: str, value: Any)"
    )

    if len(config) == 1:
        if not isinstance(config[0], dict):
            raise ValueError(err_msg)
        return ConfigContextManager(config[0])
    elif len(config) == 2:
        if not isinstance(config[0], str):
            raise ValueError(err_msg)
        return ConfigContextManager({config[0]: config[1]})
    else:
        raise ValueError(err_msg)


def compat_mode() -> kvikio.CompatMode:
    """Check if KvikIO is running in compatibility mode.

    Notice, this is not the same as the compatibility mode in cuFile. That is,
    cuFile can run in compatibility mode while KvikIO is not.

    When KvikIO is running in compatibility mode, it doesn't load `libcufile.so`.
    Instead, reads and writes are done using POSIX.

    Set the environment variable `KVIKIO_COMPAT_MODE` to enable/disable compatibility
    mode. By default, compatibility mode is enabled:

    - when `libcufile` cannot be found
    - when running in Windows Subsystem for Linux (WSL)
    - when `/run/udev` isn't readable, which typically happens when running inside
      a docker image not launched with `--volume /run/udev:/run/udev:ro`

    Returns
    -------
    bool
        Whether KvikIO is running in compatibility mode or not.
    """
    return kvikio._lib.defaults.compat_mode()


def num_threads() -> int:
    """Get the number of threads of the thread pool.

    Set the default value using `set("num_threads", value)` or by setting the
    `KVIKIO_NTHREADS` environment variable. If not set, the default value is 1.

    Returns
    -------
    nthreads: int
        The number of threads in the current thread pool.
    """
    return kvikio._lib.defaults.thread_pool_nthreads()


def task_size() -> int:
    """Get the default task size used for parallel IO operations.

    Set the default value using `set("task_size", value)` or by setting
    the `KVIKIO_TASK_SIZE` environment variable. If not set,
    the default value is 4 MiB.

    Returns
    -------
    nbytes: int
        The default task size in bytes.
    """
    return kvikio._lib.defaults.task_size()


def gds_threshold() -> int:
    """Get the default GDS threshold, which is the minimum size to use GDS.

    In order to improve performance of small IO, `.pread()` and `.pwrite()`
    implements a shortcut that circumvent the threadpool and use the POSIX
    backend directly.

    Set the default value using `set("gds_threshold", value)` or by setting the
    `KVIKIO_GDS_THRESHOLD` environment variable. If not set, the default
    value is 1 MiB.

    Returns
    -------
    nbytes : int
        The default GDS threshold size in bytes.
    """
    return kvikio._lib.defaults.gds_threshold()


def bounce_buffer_size() -> int:
    """Get the size of the bounce buffer used to stage data in host memory.

    Set the value using `set("bounce_buffer_size", value)` or by setting the
    `KVIKIO_BOUNCE_BUFFER_SIZE` environment variable. If not set, the
    value is 16 MiB.

    Returns
    -------
    nbytes : int
        The bounce buffer size in bytes.
    """
    return kvikio._lib.defaults.bounce_buffer_size()


def http_max_attempts() -> int:
    """Get the maximum number of attempts per remote IO read.

    Reads are retried up until ``http_max_attempts`` when the response has certain
    HTTP status codes.

    Set the value using `set("http_max_attempts", value)` or by setting the
    ``KVIKIO_HTTP_MAX_ATTEMPTS`` environment variable. If not set, the
    value is 3.

    Returns
    -------
    max_attempts : int
        The maximum number of remote IO reads to attempt before raising an
        error.
    """
    return kvikio._lib.defaults.http_max_attempts()


def http_status_codes() -> list[int]:
    """Get the list of HTTP status codes to retry.

    Set the value using ``set("http_status_codes", value)`` or by setting the
    ``KVIKIO_HTTP_STATUS_CODES`` environment variable. If not set, the
    default value is

    - 429
    - 500
    - 502
    - 503
    - 504

    Returns
    -------
    status_codes : list[int]
        The HTTP status codes to retry.
    """
    return kvikio._lib.defaults.http_status_codes()
