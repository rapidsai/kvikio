# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from . import _version
from .cufile import CuFile  # noqa: F401

__version__ = _version.get_versions()["version"]
