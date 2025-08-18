Zarr
====

`Zarr <https://github.com/zarr-developers/zarr-specs>`_ is a binary file format for chunked, compressed, N-Dimensional array. It is used throughout the PyData ecosystem and especially for climate and biological science applications.


`Zarr-Python <https://zarr.readthedocs.io/en/stable/>`_ is the official Python package for reading and writing Zarr arrays. Its main feature is a NumPy-like array that translates array operations into file IO seamlessly.
KvikIO provides a GPU backend to Zarr-Python that enables `GPUDirect Storage (GDS) <https://developer.nvidia.com/blog/gpudirect-storage/>`_ seamlessly.

KvikIO supports either zarr-python 2.x or zarr-python 3.x.
However, the API provided in :mod:`kvikio.zarr` differs based on which version of zarr you have, following the differences between zarr-python 2.x and zarr-python 3.x.


Zarr Python 3.x
---------------

Zarr-python includes native support for reading Zarr chunks into device memory if you `configure Zarr <https://zarr.readthedocs.io/en/stable/user-guide/gpu.html#>`__ to use GPUs.
You can use any store, but KvikIO provides :py:class:`kvikio.zarr.GDSStore` to efficiently load data directly into GPU memory.

.. code-block:: python

   >>> import zarr
   >>> from kvikio.zarr import GDSStore
   >>> zarr.config.enable_gpu()
   >>> store = GDSStore(root="data.zarr")
   >>> z = zarr.create_array(
   ...     store=store, shape=(100, 100), chunks=(10, 10), dtype="float32", overwrite=True
   ... )
   >>> type(z[:10, :10])
   cupy.ndarray
