# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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


from kvikio._lib.defaults import CompatMode  # noqa: F401
from kvikio._version import __git_commit__, __version__
from kvikio.cufile import CuFile, clear_page_cache, get_page_cache_info
from kvikio.mmap import Mmap
from kvikio.remote_file import RemoteEndpointType, RemoteFile, is_remote_file_available

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
]
