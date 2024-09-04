# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from kvikio._lib import buffer  # type: ignore


def memory_register(buf) -> None:
    """Register a device memory allocation with cuFile.

    Warning
    -------
    This API is intended for usecases where the memory is used as a streaming
    buffer that is reused across multiple cuFile IO operations.

    Parameters
    ----------
    buf: buffer-like or array-like
        Device buffer to register .
    """
    return buffer.memory_register(buf)


def memory_deregister(buf) -> None:
    """Deregister an already registered device memory from cuFile.

    Parameters
    ----------
    buf: buffer-like or array-like
        Device buffer to deregister .
    """
    buffer.memory_deregister(buf)


def bounce_buffer_free() -> int:
    """Free the host allocations used as bounce buffers.

    Returns
    -------
    Number of bytes freed.
    """
    return buffer.bounce_buffer_free()
