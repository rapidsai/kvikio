Quickstart
==========

KvikIO can be used in place of Python's built-in `open() <https://docs.python.org/3/library/functions.html#open>`_ function with the caveat that a file is always opened in binary (``"b"``) mode.
In order to open a file, use KvikIO's filehandle :py:meth:`kvikio.cufile.CuFile`.

.. code-block:: python

  import cupy
  import kvikio

  a = cupy.arange(100)
  f = kvikio.CuFile("test-file", "w")
  # Write whole array to file
  f.write(a)
  f.close()

  b = cupy.empty_like(a)
  f = kvikio.CuFile("test-file", "r")
  # Read whole array from file
  f.read(b)
  assert all(a == b)

  # Use contexmanager
  c = cupy.empty_like(a)
  with kvikio.CuFile(path, "r") as f:
      f.read(c)
  assert all(a == c)

  # Non-blocking read
  d = cupy.empty_like(a)
  with kvikio.CuFile(path, "r") as f:
      future1 = f.pread(d[:50])
      future2 = f.pread(d[50:], file_offset=d[:50].nbytes)
      future1.get()  # Wait for first read
      future2.get()  # Wait for second read
  assert all(a == d)
