# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


from . import _version
from ._lib import libcufile  # type: ignore

__version__ = _version.get_versions()["version"]


def memory_register(buf) -> None:
    return libcufile.memory_register(buf)


def memory_deregister(buf) -> None:
    libcufile.memory_deregister(buf)


class CuFile:
    """ File handle for GPUDirect Storage (GDS) """

    def __init__(self, file_path, flags="r"):
        self._handle = libcufile.CuFile(file_path, flags)

    def close(self) -> None:
        self._handle.close()

    @property
    def closed(self) -> bool:
        return self._handle.closed()

    def fileno(self) -> int:
        return self._handle.fileno()

    def open_flags(self) -> int:
        return self._handle.open_flags()

    def __enter__(self) -> "CuFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def pread(
        self, buf, size: int = None, file_offset: int = 0, ntasks=None
    ) -> libcufile.IOFuture:
        return self._handle.pread(buf, size, file_offset, ntasks)

    def pwrite(
        self, buf, size: int = None, file_offset: int = 0, ntasks=None
    ) -> libcufile.IOFuture:
        return self._handle.pwrite(buf, size, file_offset, ntasks)

    def read(self, buf, size: int = None, file_offset: int = 0, ntasks=None) -> int:
        return self.pread(buf, size, file_offset, ntasks).get()

    def write(self, buf, size: int = None, file_offset: int = 0, ntasks=None) -> int:
        return self.pwrite(buf, size, file_offset, ntasks).get()


# TODO: Wrap nicely, maybe as a dataclass?
DriverProperties = libcufile.DriverProperties
# TODO: Wrap nicely, maybe as a dataclass?
NVML = libcufile.NVML
