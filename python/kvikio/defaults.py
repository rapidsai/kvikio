# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import contextlib

from ._lib import libkvikio  # type: ignore


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

    Return
    ------
    bool
        Whether KvikIO is running in compatibility mode or not.
    """
    return libkvikio.compat_mode()


def compat_mode_reset(enable: bool) -> None:
    """Reset the compatibility mode.

    Use this function to enable/disable compatibility mode explicitly.

    Parameters
    ----------
    enable : bool
        Set to True to enable and False to disable compatibility mode
    """
    libkvikio.compat_mode_reset(enable)


@contextlib.contextmanager
def set_compat_mode(enable: bool):
    """Context for resetting the compatibility mode.

    Parameters
    ----------
    enable : bool
        Set to True to enable and False to disable compatibility mode
    """
    reset_num_threads(get_num_threads())  # Sync all running threads
    old_value = compat_mode()
    try:
        compat_mode_reset(enable)
        yield
    finally:
        compat_mode_reset(old_value)


def get_num_threads() -> int:
    """Get the number of threads of the thread pool.

    Set the default value using `reset_num_threads()` or by setting the
    `KVIKIO_NTHREADS` environment variable. If not set, the default value is 1.

    Return
    ------
    nthreads: int
        The number of threads in the current thread pool.
    """
    return libkvikio.thread_pool_nthreads()


def reset_num_threads(nthreads: int) -> None:
    """Reset the number of threads in the default thread pool.

    Waits for all currently running tasks to be completed, then destroys all threads
    in the pool and creates a new thread pool with the new number of threads. Any
    tasks that were waiting in the queue before the pool was reset will then be
    executed by the new threads. If the pool was paused before resetting it, the new
    pool will be paused as well.

    Parameters
    ----------
    nthreads : int
        The number of threads to use.
    """
    libkvikio.thread_pool_nthreads_reset(nthreads)


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
        reset_num_threads(nthreads)
        yield
    finally:
        reset_num_threads(old_value)


def task_size() -> int:
    """Get the default task size used for parallel IO operations.

    Set the default value using `task_size_reset()` or by setting
    the `KVIKIO_TASK_SIZE` environment variable. If not set,
    the default value is 16 MiB.

    Return
    ------
    nthreads: int
        The number of threads in the current thread pool.
    """
    return libkvikio.task_size()


def reset_task_size(nbytes: int) -> None:
    """Reset the default task size used for parallel IO operations.

    Parameters
    ----------
    nbytes : int
        The number of threads to use.
    """
    libkvikio.task_size_reset(nbytes)


@contextlib.contextmanager
def set_task_size(nbytes: int):
    """Context for resetting the task size used for parallel IO operations.

    Parameters
    ----------
    nbytes : int
        The number of threads to use.
    """
    old_value = task_size()
    try:
        reset_task_size(nbytes)
        yield
    finally:
        reset_task_size(old_value)
