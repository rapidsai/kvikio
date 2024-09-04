# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from kvikio._lib import driver_properties  # type: ignore
from kvikio._version import __git_commit__, __version__
from kvikio.cufile import CuFile

# TODO: Wrap nicely, maybe as a dataclass?
DriverProperties = driver_properties.DriverProperties


__all__ = ["__git_commit__", "__version__", "CuFile"]
