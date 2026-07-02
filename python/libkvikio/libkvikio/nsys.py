# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def nsys_plugin_search_dir() -> str:
    """Return the directory that holds KvikIO's bundled Nsight Systems plugins.

    When libkvikio is installed as a wheel, the ``kvikio_nic`` plugin (an executable
    plus its ``nsys-plugin.yaml`` manifest) is packaged under this directory. Point
    Nsight Systems at it by exporting the returned path on ``NSYS_PLUGIN_SEARCH_DIRS``,
    for example::

        export NSYS_PLUGIN_SEARCH_DIRS="$(python -c 'import libkvikio; print(libkvikio.nsys_plugin_search_dir())')"
        nsys profile --enable=kvikio_nic ...

    Returns
    -------
    str
        The ``share/kvikio/nsys-plugins`` directory inside the installed
        ``libkvikio`` package. The path is returned even if it does not exist, for
        example when the plugin was not built.
    """
    return os.path.join(os.path.dirname(__file__), "share", "kvikio", "nsys-plugins")
