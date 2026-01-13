# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvikio._lib import stream as stream_module  # type: ignore


def stream_register(raw_stream: int, flags: int) -> None:
    """Registers the CUDA stream to the cuFile subsystem.

    Parameters
    ----------
    raw_stream: int
        Raw CUDA stream which queues the async I/O operations
    flags: int
        Specifies when the I/O parameters become valid (submission time or execution
        time) and what I/O parameters are page-aligned. For details, refer to
        https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilestreamregister
    """
    stream_module.stream_register(raw_stream, flags)


def stream_deregister(raw_stream: int) -> None:
    """Deregisters the CUDA stream from the cuFile subsystem.

    Parameters
    ----------
    raw_stream: int
        Raw CUDA stream which queues the async I/O operations
    """
    stream_module.stream_deregister(raw_stream)
