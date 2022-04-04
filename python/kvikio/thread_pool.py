# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


from ._lib import libkvikio  # type: ignore


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
        The number of threads to use. The default value can be specified by setting
        the `KVIKIO_NTHREADS` environment variable. If not set, the default value
        is 1.
    """
    libkvikio.thread_pool_reset_num_threads(nthreads)


def get_num_threads() -> int:
    """Get the number of threads of the thread pool.

    Return
    ------
    nthreads: int
        The number of threads in the current thread pool.
    """
    return libkvikio.thread_pool_get_num_threads()
