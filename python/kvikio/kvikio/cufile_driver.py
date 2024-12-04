# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import atexit
from typing import Tuple

from kvikio._lib import cufile_driver  # type: ignore

# TODO: Wrap nicely, maybe as a dataclass?
# <https://github.com/rapidsai/kvikio/issues/526>
DriverProperties = cufile_driver.DriverProperties


def libcufile_version() -> Tuple[int, int]:
    """Get the libcufile version.

    Returns (0, 0) for cuFile versions prior to v1.8.

    Notes
    -----
    This is not the version of the CUDA toolkit. cufile is part of the
    toolkit but follows its own version scheme.

    Returns
    -------
    The version as a tuple (MAJOR, MINOR).
    """
    v = cufile_driver.libcufile_version()
    major = v // 1000
    minor = (v % 1000) // 10
    return (major, minor)


def driver_open() -> None:
    """Open the cuFile driver

    cuFile accepts multiple calls to `driver_open()`. Only the first call
    opens the driver, but every call must have a matching call to
    `driver_close()`.

    Normally, it is not required to open and close the cuFile driver since
    it is done automatically.

    Raises
    ------
    RuntimeError
        If cuFile isn't available.
    """
    return cufile_driver.driver_open()


def driver_close() -> None:
    """Close the cuFile driver

    cuFile accepts multiple calls to `driver_open()`. Only the first call
    opens the driver, but every call must have a matching call to
    `driver_close()`.

    Raises
    ------
    RuntimeError
        If cuFile isn't available.
    """
    return cufile_driver.driver_close()


def initialize() -> None:
    """Open the cuFile driver and close it again at module exit

    Normally, it is not required to open and close the cuFile driver since
    it is done automatically.

    Notes
    -----
    Registers an atexit handler that calls :func:`driver_close`.

    Raises
    ------
    RuntimeError
        If cuFile isn't available.
    """
    driver_open()
    atexit.register(driver_close)
