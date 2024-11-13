Zarr
====

`Zarr <https://github.com/zarr-developers/zarr-specs>`_ is a binary file format for chunked, compressed, N-Dimensional array. It is used throughout the PyData ecosystem and especially for climate and biological science applications.


`Zarr-Python <https://zarr.readthedocs.io/en/stable/>`_ is the official Python package for reading and writing Zarr arrays. Its main feature is a NumPy-like array that translates array operations into file IO seamlessly.
KvikIO provides a GPU backend to Zarr-Python that enables `GPUDirect Storage (GDS) <https://developer.nvidia.com/blog/gpudirect-storage/>`_ seamlessly.

The following is an example of how to use the convenience function :py:meth:`kvikio.zarr.open_cupy_array`
to create a new Zarr array and how to open an existing Zarr array.


.. literalinclude:: ../../python/kvikio/examples/zarr_cupy_nvcomp.py
    :language: python
