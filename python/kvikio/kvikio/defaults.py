# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import contextlib

import kvikio._lib.defaults


def compat_mode() -> bool:
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


def compat_mode_reset(compat_mode: bool | str) -> None:
    """Reset the compatibility mode.

    Use this function to enable/disable compatibility mode explicitly.

    Parameters
    ----------
    enable : bool
        Set to True to enable and False to disable compatibility mode
    """
    if isinstance(compat_mode, bool):
        kvikio._lib.defaults.compat_mode_reset_bool(compat_mode)
    else:
        kvikio._lib.defaults.compat_mode_reset_str(compat_mode)


@contextlib.contextmanager
def set_compat_mode(enable: bool):
    """Context for resetting the compatibility mode.

    Parameters
    ----------
    enable : bool
        Set to True to enable and False to disable compatibility mode
    """
    num_threads_reset(get_num_threads())  # Sync all running threads
    old_value = compat_mode()
    try:
        compat_mode_reset(enable)
        yield
    finally:
        compat_mode_reset(old_value)


def get_num_threads() -> int:
    """Get the number of threads of the thread pool.

    Set the default value using `num_threads_reset()` or by setting the
    `KVIKIO_NTHREADS` environment variable. If not set, the default value is 1.

    Returns
    -------
    nthreads: int
        The number of threads in the current thread pool.
    """
    return kvikio._lib.defaults.thread_pool_nthreads()


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
    kvikio._lib.defaults.thread_pool_nthreads_reset(nthreads)


@contextlib.contextmanager
def set_num_threads(nthreads: int):
    """Context for resetting the number of threads in the default thread pool.

    Parameters
    ----------
    nthreads : int
        The number of threads to use.
    """
    old_value = get_num_threads()
    try:
        num_threads_reset(nthreads)
        yield
    finally:
        num_threads_reset(old_value)


def task_size() -> int:
    """Get the default task size used for parallel IO operations.

    Set the default value using `task_size_reset()` or by setting
    the `KVIKIO_TASK_SIZE` environment variable. If not set,
    the default value is 4 MiB.

    Returns
    -------
    nbytes: int
        The default task size in bytes.
    """
    return kvikio._lib.defaults.task_size()


def task_size_reset(nbytes: int) -> None:
    """Reset the default task size used for parallel IO operations.

    Parameters
    ----------
    nbytes : int
        The default task size in bytes.
    """
    kvikio._lib.defaults.task_size_reset(nbytes)


@contextlib.contextmanager
def set_task_size(nbytes: int):
    """Context for resetting the task size used for parallel IO operations.

    Parameters
    ----------
    nbytes : int
        The default task size in bytes.
    """
    old_value = task_size()
    try:
        task_size_reset(nbytes)
        yield
    finally:
        task_size_reset(old_value)


def gds_threshold() -> int:
    """Get the default GDS threshold, which is the minimum size to use GDS.

    In order to improve performance of small IO, `.pread()` and `.pwrite()`
    implements a shortcut that circumvent the threadpool and use the POSIX
    backend directly.

    Set the default value using `gds_threshold_reset()` or by setting the
    `KVIKIO_GDS_THRESHOLD` environment variable. If not set, the default
    value is 1 MiB.

    Returns
    -------
    nbytes : int
        The default GDS threshold size in bytes.
    """
    return kvikio._lib.defaults.gds_threshold()


def gds_threshold_reset(nbytes: int) -> None:
    """Reset the default GDS threshold, which is the minimum size to use GDS.

    Parameters
    ----------
    nbytes : int
        The default GDS threshold size in bytes.
    """
    kvikio._lib.defaults.gds_threshold_reset(nbytes)


@contextlib.contextmanager
def set_gds_threshold(nbytes: int):
    """Context for resetting the default GDS threshold.

    Parameters
    ----------
    nbytes : int
        The default GDS threshold size in bytes.
    """
    old_value = gds_threshold()
    try:
        gds_threshold_reset(nbytes)
        yield
    finally:
        gds_threshold_reset(old_value)


def bounce_buffer_size() -> int:
    """Get the size of the bounce buffer used to stage data in host memory.

    Set the value using `bounce_buffer_size_reset()` or by setting the
    `KVIKIO_BOUNCE_BUFFER_SIZE` environment variable. If not set, the
    value is 16 MiB.

    Return
    ------
    nbytes : int
        The bounce buffer size in bytes.
    """
    return kvikio._lib.defaults.bounce_buffer_size()


def bounce_buffer_size_reset(nbytes: int) -> None:
    """Reset the size of the bounce buffer used to stage data in host memory.

    Parameters
    ----------
    nbytes : int
        The bounce buffer size in bytes.
    """
    kvikio._lib.defaults.bounce_buffer_size_reset(nbytes)


@contextlib.contextmanager
def set_bounce_buffer_size(nbytes: int):
    """Context for resetting the the size of the bounce buffer.

    Parameters
    ----------
    nbytes : int
        The bounce buffer size in bytes.
    """
    old_value = bounce_buffer_size()
    try:
        bounce_buffer_size_reset(nbytes)
        yield
    finally:
        bounce_buffer_size_reset(old_value)
