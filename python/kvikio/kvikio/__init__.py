# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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


from kvikio._version import __git_commit__, __version__
from kvikio.cufile import CuFile
from kvikio.remote_file import RemoteFile, is_remote_file_available
from kvikio._lib.defaults import CompatMode

__all__ = [
    "__git_commit__",
    "__version__",
    "CuFile",
    "RemoteFile",
    "is_remote_file_available",
]
