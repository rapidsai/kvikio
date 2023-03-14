# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import cunumeric as num
import legate_kvikio
import legate_kvikio.core


def main(path):

    src = num.arange(10_000_000)
    written = legate_kvikio.core.write(path=path, obj=src)

    # Blocking
    written = legate_kvikio.core.get_written_nbytes(written)
    assert written == src.nbytes

    dst = num.empty_like(src)
    legate_kvikio.core.read(path=path, obj=dst)
    assert (src == dst).all()


if __name__ == "__main__":
    main("/tmp/kvikio-legate-basic-io")
