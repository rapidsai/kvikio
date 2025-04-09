# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from importlib import metadata as _metadata

from packaging.version import Version as _Version, parse as _parse

if _parse(_metadata.version("zarr")) >= _Version("3.0.0"):
    from ._zarr_python_3 import *  # noqa: F401,F403
else:
    from ._zarr_python_2 import *  # type: ignore[assignment] # noqa: F401,F403
