# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvikio._lib import buffer  # type: ignore


def memory_register(buf) -> None:
    """Register a device memory allocation with cuFile for GPUDirect Storage access.

    This function automatically discovers the base address and size of the CUDA memory
    allocation containing ``buf``. The entire underlying allocation is registered,
    regardless of which portion ``buf`` points to.

    Registration pins the memory for GPU Direct DMA transfers, which can improve
    performance when the same buffer is reused across multiple cuFile I/O operations.

    In compatibility mode (when GDS is unavailable), this function is a no-op.

    Warning
    -------
    This API is intended for streaming buffers reused across multiple cuFile I/O
    operations. For one-time transfers, the overhead of registration may outweigh the
    benefits.

    Parameters
    ----------
    buf: buffer-like or array-like
        Device buffer to register .
    """
    return buffer.memory_register(buf)


def memory_deregister(buf) -> None:
    """Deregister a device memory allocation from cuFile.

    This function automatically discovers the base address of the CUDA memory
    allocation containing ``buf``. The entire underlying allocation is deregistered,
    regardless of which portion ``buf`` points to.

    In compatibility mode (when GDS is unavailable), this function is a no-op.

    Parameters
    ----------
    buf: buffer-like or array-like
        Device buffer to deregister.
    """
    buffer.memory_deregister(buf)


def bounce_buffer_free() -> int:
    """Free the host allocations used as bounce buffers.

    Returns
    -------
    Number of bytes freed.
    """
    return buffer.bounce_buffer_free()
