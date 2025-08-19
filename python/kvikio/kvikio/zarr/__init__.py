# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

try:
    from ._zarr_python_3 import GDSStore
except ImportError as e:
    raise ImportError("kvikio.zarr requires the optional 'zarr>=3' dependency") from e

__all__ = ["GDSStore"]
