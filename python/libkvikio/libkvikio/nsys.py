# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def nsys_plugin_search_dir() -> str:
    """Return the directory that holds KvikIO's bundled Nsight Systems plugins. Valid
    only for pip installation.

    Returns
    -------
    str
        The ``share/kvikio/nsys-plugins`` directory inside the installed ``libkvikio``
        package.
    """
    return os.path.join(os.path.dirname(__file__), "share", "kvikio", "nsys-plugins")
