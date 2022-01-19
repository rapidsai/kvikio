# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import errno
import os
import os.path
import shutil
import uuid

import cupy
import zarr.storage
from zarr.util import retry_call

import kvikio
from kvikio._lib.arr import asarray


class GDSStore(zarr.storage.DirectoryStore):
    """GPUDirect Storage (GDS) class using directories and files.

    This class works like `zarr.storage.DirectoryStore` but use GPU
    buffers and will use GDS when applicable.
    The store supports both CPU and GPU buffers but when reading, GPU
    buffers are returned always.

    TODO: Write metadata to disk in order to perserve the item types such that
    GPU items are read as GPU device buffers and CPU items are read as bytes.
    """

    def __eq__(self, other):
        return isinstance(other, GDSStore) and self.path == other.path

    def _fromfile(self, fn):
        """Read `fn` into device memory _unless_ `fn` refers to Zarr metadata"""
        if os.path.basename(fn) in [
            zarr.storage.array_meta_key,
            zarr.storage.group_meta_key,
            zarr.storage.attrs_key,
        ]:
            return super()._fromfile(fn)
        else:
            nbytes = os.path.getsize(fn)
            with kvikio.CuFile(fn, "r") as f:
                ret = cupy.empty(nbytes, dtype="u1")
                read = f.read(ret)
                assert read == nbytes
                return ret

    def _tofile(self, a, fn):
        a = asarray(a)
        assert a.contiguous
        if a.cuda:
            with kvikio.CuFile(fn, "w") as f:
                written = f.write(a)
                assert written == a.nbytes
        else:
            super()._tofile(a.obj, fn)

    def __setitem__(self, key, value):
        """
        We have to overwrite this because `DirectoryStore.__setitem__`
        converts `value` to a NumPy array always
        """
        key = self._normalize_key(key)

        # coerce to flat, contiguous buffer (ideally without copying)
        arr = asarray(value)
        if arr.contiguous:
            value = arr
        else:
            if arr.cuda:
                # value = cupy.ascontiguousarray(value)
                value = arr.reshape(-1, order="A")
            else:
                # can flatten without copy
                value = arr.reshape(-1, order="A")

        # destination path for key
        file_path = os.path.join(self.path, key)

        # ensure there is no directory in the way
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)

        # ensure containing directory exists
        dir_path, file_name = os.path.split(file_path)
        if os.path.isfile(dir_path):
            raise KeyError(key)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise KeyError(key)

        # write to temporary file
        # note we're not using tempfile.NamedTemporaryFile to avoid
        # restrictive file permissions
        temp_name = file_name + "." + uuid.uuid4().hex + ".partial"
        temp_path = os.path.join(dir_path, temp_name)
        try:
            self._tofile(value, temp_path)

            # move temporary file into place;
            # make several attempts at writing the temporary file to get past
            # potential antivirus file locking issues
            retry_call(
                os.replace, (temp_path, file_path), exceptions=(PermissionError,)
            )
        finally:
            # clean up if temp file still exists for whatever reason
            if os.path.exists(temp_path):  # pragma: no cover
                os.remove(temp_path)
