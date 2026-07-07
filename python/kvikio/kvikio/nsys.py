# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys


def nsys_plugin_search_dir() -> str:
    """Return the directory that holds KvikIO's bundled Nsight Systems plugins.

    Works for both installation channels, which lay out the plugin differently.

    - pip (inside the ``libkvikio`` wheel, a Python package):

    .. code-block:: text

       site-packages/libkvikio/
       |-- __init__.py, load.py, nsys.py
       |-- lib/libkvikio.so
       |-- share/kvikio/nsys-plugins/ --> returned
           |-- kvikio_nic/{kvikio_nic_nsys_plugin, nsys-plugin.yaml}

    - conda (the ``libkvikio`` conda package, C++ files only, no Python module):

    .. code-block:: text

       $CONDA_PREFIX/
       |-- lib/libkvikio.so
       |-- include/kvikio/
       |-- share/kvikio/nsys-plugins/ --> returned
           |-- kvikio_nic/{kvikio_nic_nsys_plugin, nsys-plugin.yaml}

    See :doc:`profiling </profiling>` for how to enable the plugin in Nsight Systems.

    Returns
    -------
    str
        The plugin search directory. The path is returned even if it does not exist,
        for example when the plugin was not built.
    """
    try:
        import libkvikio

        # pip: the plugin ships inside the libkvikio wheel.
        return libkvikio.nsys_plugin_search_dir()
    except ModuleNotFoundError:
        # conda: the libkvikio conda package installs the plugin into the
        # environment prefix, which is sys.prefix in a conda environment.
        return os.path.join(sys.prefix, "share", "kvikio", "nsys-plugins")
