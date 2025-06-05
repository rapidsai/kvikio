# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

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
from kvikio.cufile import CuFile, get_page_cache_info
from kvikio.remote_file import RemoteFile, is_remote_file_available

__all__ = [
    "__git_commit__",
    "__version__",
    "CuFile",
    "get_page_cache_info",
    "RemoteFile",
    "is_remote_file_available",
]
