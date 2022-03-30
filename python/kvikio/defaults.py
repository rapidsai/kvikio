# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


from ._lib import libkvikio  # type: ignore


def compat_mode() -> int:
    """ Check if KvikIO is running in compatibility mode.

    Return
    ------
    bool
        Whether KvikIO is running in compatibility mode or not.
    """
    return libkvikio.compat_mode()
