# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# If libkvikio was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libkvikio
except ModuleNotFoundError:
    pass
else:
    libkvikio.load_library()
    del libkvikio


from kvikio._lib.defaults import CompatMode, RemoteBackendType  # noqa: F401
from kvikio._version import __git_commit__, __version__
from kvikio.buffer import bounce_buffer_free, memory_deregister, memory_register
from kvikio.cufile import CuFile, clear_page_cache, get_page_cache_info
from kvikio.mmap import Mmap
from kvikio.remote_file import RemoteEndpointType, RemoteFile, is_remote_file_available
from kvikio.stream import stream_deregister, stream_register

__all__ = [
    "__git_commit__",
    "__version__",
    "clear_page_cache",
    "CuFile",
    "Mmap",
    "get_page_cache_info",
    "is_remote_file_available",
    "RemoteEndpointType",
    "RemoteFile",
    "stream_register",
    "stream_deregister",
    "memory_register",
    "memory_deregister",
    "bounce_buffer_free",
]
