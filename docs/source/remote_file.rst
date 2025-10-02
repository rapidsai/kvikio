Remote File
===========

KvikIO provides direct access to remote files, including AWS S3, WebHDFS, and generic HTTP/HTTPS.

Example
-------

.. literalinclude:: ../../python/kvikio/examples/http_io.py
    :language: python

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
