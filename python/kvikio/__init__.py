# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from ._lib import libkvikio  # type: ignore
from .cufile import CuFile  # noqa: F401

__version__ = "24.02.01"


def memory_register(buf) -> None:
    return libkvikio.memory_register(buf)


def memory_deregister(buf) -> None:
    libkvikio.memory_deregister(buf)


# TODO: Wrap nicely, maybe as a dataclass?
DriverProperties = libkvikio.DriverProperties
