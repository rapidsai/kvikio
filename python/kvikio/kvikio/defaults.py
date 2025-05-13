# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


from typing import Any, overload

import kvikio._lib.defaults
from kvikio.utils import call_once, kvikio_deprecation_notice


class ConfigContextManager:
    """Context manager allowing the KvikIO configurations to be set upon entering a
    `with` block, and automatically reset upon leaving the block.
    """

    def __init__(self, config: dict[str, str]):
        (
            self._property_getters,
            self._property_setters,
        ) = self._property_getter_and_setter()
        self._old_properties = {}

        for key, value in config.items():
            self._old_properties[key] = self._get_property(key)
            self._set_property(key, value)

    def __enter__(self):
        return None

    def __exit__(self, type_unused, value, traceback_unused):
        for key, value in self._old_properties.items():
            self._set_property(key, value)

    def _get_property(self, property: str) -> Any:
        if property == "num_threads":
            property = "thread_pool_nthreads"
        func = self._property_getters[property]
        return func()

    def _set_property(self, property: str, value: Any):
        if property == "num_threads":
            property = "thread_pool_nthreads"
        func = self._property_setters[property]
        func(value)

    @call_once
    def _property_getter_and_setter(self) -> tuple[dict[str, Any], dict[str, Any]]:
        module_dict = vars(kvikio._lib.defaults)

        property_getter_names = [
            "compat_mode",
            "thread_pool_nthreads",
            "task_size",
            "gds_threshold",
            "bounce_buffer_size",
            "http_max_attempts",
            "http_status_codes",
            "http_timeout",
        ]

        property_getters = {}
        property_setters = {}

        for name in property_getter_names:
            property_getters[name] = module_dict[name]
            property_setters[name] = module_dict["set_" + name]
        return property_getters, property_setters


@overload
def set(config: dict[str, Any], /) -> ConfigContextManager:
    ...


@overload
def set(key: str, value: Any, /) -> ConfigContextManager:
    ...


def set(*config) -> ConfigContextManager:
    """Set KvikIO configurations.

    Examples:

    - To set one or more properties

      .. code-block:: python

         # Set the property globally.
         kvikio.defaults.set({"prop1": value1, "prop2": value2})

         # Set the property with a context manager.
         # The property automatically reverts to its old value
         # after leaving the `with` block.
         with kvikio.defaults.set({"prop1": value1, "prop2": value2}):
             ...

    - To set a single property

      .. code-block:: python

         # Set the property globally.
         kvikio.defaults.set("prop", value)

         # Set the property with a context manager.
         # The property automatically reverts to its old value
         # after leaving the `with` block.
         with kvikio.defaults.set("prop", value):
             ...

    Parameters
    ----------
    config
        The configurations. Can either be a single parameter (dict) consisting of one
        or more properties, or two parameters key (string) and value (Any)
        indicating a single property.

        Valid configuration names are:

        - ``"compat_mode"``
        - ``"num_threads"``
        - ``"task_size"``
        - ``"gds_threshold"``
        - ``"bounce_buffer_size"``
        - ``"http_max_attempts"``
        - ``"http_status_codes"``
        - ``*http_timeout*``

    Returns
    -------
    ConfigContextManager
       A context manager. If used in a `with` statement, the configuration will revert
       to its old value upon leaving the block.
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


def get(config_name: str) -> Any:
    """Get KvikIO configurations.

    Parameters
    ----------
    config_name: str
        The name of the configuration.

        Valid configuration names are:

        - ``"compat_mode"``
        - ``"num_threads"``
        - ``"task_size"``
        - ``"gds_threshold"``
        - ``"bounce_buffer_size"``
        - ``"http_max_attempts"``
        - ``"http_status_codes"``

    Returns
    -------
    Any
        The value of the configuration.
    """
    context_manager = ConfigContextManager({})
    return context_manager._get_property(config_name)


def is_compat_mode_preferred() -> bool:
    return kvikio._lib.defaults.is_compat_mode_preferred()


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.get()` instead.", since="25.04")
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


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.get()` instead.", since="25.04")
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


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.get()` instead.", since="25.04")
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


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.get()` instead.", since="25.04")
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


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.get()` instead.", since="25.04")
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


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.get()` instead.", since="25.04")
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


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.get()` instead.", since="25.04")
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


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def compat_mode_reset(compatmode: kvikio.CompatMode) -> None:
    """Reset the compatibility mode.

    Use this function to enable/disable compatibility mode explicitly.

    Parameters
    ----------
    compatmode : kvikio.CompatMode
        Set to kvikio.CompatMode.ON to enable and kvikio.CompatMode.OFF to disable
        compatibility mode, or kvikio.CompatMode.AUTO to let KvikIO determine: try
        OFF first, and upon failure, fall back to ON.
    """
    set("compat_mode", compatmode)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def set_compat_mode(compatmode: kvikio.CompatMode):
    """Same with compat_mode_reset."""
    compat_mode_reset(compatmode)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def num_threads_reset(nthreads: int) -> None:
    """Reset the number of threads in the default thread pool.

    Waits for all currently running tasks to be completed, then destroys all threads
    in the pool and creates a new thread pool with the new number of threads. Any
    tasks that were waiting in the queue before the pool was reset will then be
    executed by the new threads. If the pool was paused before resetting it, the new
    pool will be paused as well.

    Parameters
    ----------
    nthreads : int
        The number of threads to use. The default value can be specified by setting
        the `KVIKIO_NTHREADS` environment variable. If not set, the default value
        is 1.
    """
    set("num_threads", nthreads)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def set_num_threads(nthreads: int):
    """Same with num_threads_reset."""
    set("num_threads", nthreads)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def task_size_reset(nbytes: int) -> None:
    """Reset the default task size used for parallel IO operations.

    Parameters
    ----------
    nbytes : int
        The default task size in bytes.
    """
    set("task_size", nbytes)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def set_task_size(nbytes: int):
    """Same with task_size_reset."""
    set("task_size", nbytes)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def gds_threshold_reset(nbytes: int) -> None:
    """Reset the default GDS threshold, which is the minimum size to
    use GDS.

    Parameters
    ----------
    nbytes : int
        The default GDS threshold size in bytes.
    """
    set("gds_threshold", nbytes)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def set_gds_threshold(nbytes: int):
    """Same with gds_threshold_reset."""
    set("gds_threshold", nbytes)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def bounce_buffer_size_reset(nbytes: int) -> None:
    """Reset the size of the bounce buffer used to stage data in host
    memory.

    Parameters
    ----------
    nbytes : int
        The bounce buffer size in bytes.
    """
    set("bounce_buffer_size", nbytes)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def set_bounce_buffer_size(nbytes: int):
    """Same with bounce_buffer_size_reset."""
    set("bounce_buffer_size", nbytes)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def http_max_attempts_reset(attempts: int) -> None:
    """Reset the maximum number of attempts per remote IO read.

    Parameters
    ----------
    attempts : int
        The maximum number of attempts to try before raising an error.
    """
    set("http_max_attempts", attempts)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def set_http_max_attempts(attempts: int):
    """Same with http_max_attempts_reset."""
    set("http_max_attempts", attempts)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def http_status_codes_reset(status_codes: list[int]) -> None:
    """Reset the list of HTTP status codes to retry.

    Parameters
    ----------
    status_codes : list[int]
        The HTTP status codes to retry.
    """
    set("http_status_codes", status_codes)


@kvikio_deprecation_notice("Use :meth:`kvikio.defaults.set()` instead.", since="25.04")
def set_http_status_codes(status_codes: list[int]):
    """Same with http_status_codes_reset."""
    set("http_status_codes", status_codes)
