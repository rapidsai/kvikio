Zarr
====

KvikIO implements a Zarr-Python backend for reading and writing GPU data to file seamlessly.

The following is an example of how to use the convenience function :py:meth:`kvikio.zarr.open_cupy_array`
to create a new Zarr array and how open an existing Zarr array.


.. literalinclude:: ../../python/examples/zarr_cupy_nvcomp.py
    :language: python
