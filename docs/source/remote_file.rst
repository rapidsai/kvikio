Remote File
===========

KvikIO provides direct access to remote files, including AWS S3, WebHDFS, and generic HTTP/HTTPS.

Example
-------

.. literalinclude:: ../../python/kvikio/examples/http_io.py
    :language: python

S3-over-RDMA (NVIDIA cuObject)
------------------------------

For RDMA-capable S3 endpoints (such as MinIO AIStor), KvikIO can transfer object payloads directly
into the destination buffer over RDMA using NVIDIA cuObject, bypassing the HTTP body and the CPU copy.
This is an opt-in data plane for the S3 endpoint:

- Build KvikIO on a host where the cuObject SDK (``cuobjclient.h`` and ``libcuobjclient``, shipped with
  the CUDA Toolkit) is present. CMake then builds ``libkvikio_cuobj_shim.so`` automatically.
- At runtime, set ``KVIKIO_REMOTE_RDMA=on``. If the shim cannot be found on the loader path, point
  ``KVIKIO_CUOBJ_SHIM`` at it. cuObject also needs a cuFile/cuObject JSON config describing the RDMA NIC
  (``CUFILE_ENV_PATH_JSON``).

When enabled and a cuObject connection is established, ``RemoteHandle.read``/``pread`` register the
destination buffer (host or device memory) with cuObject and issue a body-less range request carrying
the signed ``x-amz-rdma-token`` header; the endpoint RDMA-writes the payload into the buffer. If the
endpoint does not honor the request (``x-amz-rdma-reply: 501``), the read fails rather than silently
falling back, so only enable it against an RDMA-capable endpoint.

AWS S3 object naming requirement
--------------------------------

KvikIO imposes the following naming requirements derived from the `AWS object naming guidelines <https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html>`_ .

 - ``!``, ``*``, ``'``, ``(``, ``)``, ``&``, ``$``, ``@``, ``=``, ``;``, ``:``, ``+``, ``,``: These special characters are automatically encoded by KvikIO, and are safe for use in key names.

 - ``-``, ``_``, ``.``: These special characters are **not** automatically encoded by KvikIO, but are still safe for use in key names.

 - ``/`` is used as path separator and must not appear in the object name itself.

 - Space character must be explicitly encoded (``%20``) because it will otherwise render the URL malformed.

 - ``?`` must be explicitly encoded (``%3F``) because it will otherwise cause ambiguity with the query string.

 - Control characters ``0x00`` ~ ``0x1F`` hexadecimal (0~31 decimal) and ``0x7F`` (127) are automatically encoded by KvikIO, and are safe for use in key names.

 - Other printable special characters must be avoided, such as ``\``, ``{``, ``^``, ``}``, ``%``, `````, ``]``, ``"``, ``>``, ``[``, ``~``, ``<``, ``#``, ``|``.

 - Non-ASCII characters ``0x80`` ~ ``0xFF`` (128~255) must be avoided.
