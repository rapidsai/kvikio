# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import argparse
import subprocess


def drop_vm_cache(args: argparse.Namespace) -> None:
    """Tells the Linux kernel to drop the page, inode, and dentry caches

    See <https://linux-mm.org/Drop_Caches>

    Parameters
    ----------
    args
        The parsed command line arguments.
    """
    if args.drop_vm_cache:
        subprocess.check_output(["sudo /sbin/sysctl vm.drop_caches=3"], shell=True)


