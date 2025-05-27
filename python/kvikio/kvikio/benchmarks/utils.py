# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import annotations

import argparse
import contextlib
import os
import os.path
import pathlib
import subprocess

from dask.utils import format_bytes

import kvikio
import kvikio.cufile_driver
import kvikio.defaults

has_pynvml = False
with contextlib.suppress(ImportError):
    import pynvml

    has_pynvml = True


def drop_vm_cache() -> None:
    """Tells the Linux kernel to drop the page, inode, and dentry caches

    See <https://linux-mm.org/Drop_Caches>
    """
    subprocess.check_output(["sudo /sbin/sysctl vm.drop_caches=3"], shell=True)


def pprint_sys_info() -> None:
    """Pretty print system information"""

    version = kvikio.cufile_driver.libcufile_version()
    props = kvikio.cufile_driver.properties

    gpu_name = mem_total = bar1_total = "Unknown (install nvidia-ml-py)"
    if has_pynvml:
        dev = None
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlInit()
            dev = pynvml.nvmlDeviceGetHandleByIndex(0)

        if dev is not None:
            gpu_name = f"{pynvml.nvmlDeviceGetName(dev)} (dev #0)"
            try:
                mem_total = format_bytes(pynvml.nvmlDeviceGetMemoryInfo(dev).total)
            except pynvml.NVMLError_NotSupported:
                mem_total = "Device has no memory resource"
            try:
                bar1_total = pynvml.nvmlDeviceGetBAR1MemoryInfo(dev).bar1Total
            except pynvml.NVMLError_NotSupported:
                bar1_total = "Device has no BAR1 support"

    if version == (0, 0):
        libcufile_version = "unknown (earlier than cuFile 1.8)"
    else:
        libcufile_version = f"{version[0]}.{version[1]}"
    gds_version = "N/A (Compatibility Mode)"
    if props.is_gds_available:
        gds_version = f"v{props.major_version}.{props.minor_version}"
    gds_config_json_path = os.path.realpath(
        os.getenv("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")
    )

    if kvikio.defaults.compat_mode():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("   WARNING - KvikIO compat mode   ")
        print("      libcufile.so not used       ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif not props.is_gds_available:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("   WARNING - cuFile compat mode   ")
        print("         GDS not enabled          ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"GPU               | {gpu_name}")
    print(f"GPU Memory Total  | {mem_total}")
    print(f"BAR1 Memory Total | {bar1_total}")
    print(f"libcufile version | {libcufile_version}")
    print(f"GDS driver        | {gds_version}")
    print(f"GDS config.json   | {gds_config_json_path}")


def parse_directory(x: str | None) -> pathlib.Path | None:
    """Given an argparse argument, return a dir path.

    None are passed through untouched.
    Raise argparse.ArgumentTypeError if `x` isn't a directory (or None).

    Parameters
    ----------
    x
        argparse argument

    Returns
    -------
    The directory path or None
    """
    if x is None:
        return x
    else:
        p = pathlib.Path(x)
        if not p.is_dir():
            raise argparse.ArgumentTypeError("Must be a directory")
        return p
