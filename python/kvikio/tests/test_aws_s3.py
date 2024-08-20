# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import multiprocessing as mp
import os
import socket
import time
from contextlib import contextmanager

import pytest

import kvikio

# TODO: remove before PR merge. Trigger CI error if the remote module wasn't built
import kvikio._lib.remote_handle  # isort: skip

pytestmark = pytest.mark.skipif(
    not kvikio.is_remote_file_available(),
    reason="cannot test remote IO, please build KvikIO with with AWS S3 support",
)

# Notice, we import boto and moto after the `is_remote_file_available` check.
import boto3  # noqa: E402
import moto  # noqa: E402
import moto.server  # noqa: E402


@pytest.fixture(scope="session")
def endpoint_ip():
    return "127.0.0.1"


@pytest.fixture(scope="session")
def endpoint_port():
    # Return a free port per worker session.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@contextmanager
def ensure_safe_environment_variables():
    """
    Get a context manager to safely set environment variables
    All changes will be undone on close, hence environment variables set
    within this contextmanager will neither persist nor change global state.
    """
    saved_environ = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)


def start_s3_server(ip_address, port):
    server = moto.server.ThreadedMotoServer(ip_address=ip_address, port=port)
    server.start()
    time.sleep(180)
    print("ThreadedMotoServer shutting down because of timeout (180s)")


@pytest.fixture(scope="session")
def s3_base(endpoint_ip, endpoint_port):
    """
    Fixture to set up moto server in separate process
    """
    with ensure_safe_environment_variables():
        # Use fake aws credentials
        os.environ["AWS_ACCESS_KEY_ID"] = "foobar_key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foobar_secret"
        os.environ["AWS_SECURITY_TOKEN"] = "foobar_security_token"
        os.environ["AWS_SESSION_TOKEN"] = "foobar_session_token"
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        os.environ["AWS_ENDPOINT_URL"] = f"http://{endpoint_ip}:{endpoint_port}"

        p = mp.Process(target=start_s3_server, args=(endpoint_ip, endpoint_port))
        p.start()
        yield os.environ["AWS_ENDPOINT_URL"]
        p.kill()


@contextmanager
def s3_context(s3_base, bucket, files=None):
    if files is None:
        files = {}
    with ensure_safe_environment_variables():
        client = boto3.client("s3", endpoint_url=s3_base)
        client.create_bucket(Bucket=bucket, ACL="public-read-write")
        for f, data in files.items():
            client.put_object(Bucket=bucket, Key=f, Body=data)
        yield
        for f, data in files.items():
            try:
                client.delete_object(Bucket=bucket, Key=f)
            except Exception:
                pass


def test_read(s3_base, xp):
    bucket_name = "test_read"
    object_name = "a1"
    a = xp.arange(10_000_000)
    with s3_context(s3_base=s3_base, bucket=bucket_name, files={object_name: bytes(a)}):
        with kvikio.RemoteFile(bucket_name, object_name) as f:
            assert f.nbytes() == a.nbytes
            b = xp.empty_like(a)
            assert f.read(buf=b) == a.nbytes
            xp.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "start,end",
    [
        (0, 10 * 4096),
        (1, int(1.3 * 4096)),
        (int(2.1 * 4096), int(5.6 * 4096)),
        (42, int(2**23)),
    ],
)
def test_read_with_file_offset(s3_base, xp, start, end):
    bucket_name = "test_read"
    object_name = "a1"
    a = xp.arange(end, dtype=xp.int64)
    with s3_context(s3_base=s3_base, bucket=bucket_name, files={object_name: bytes(a)}):
        with kvikio.RemoteFile(bucket_name, object_name) as f:
            b = xp.zeros(shape=(end - start,), dtype=xp.int64)
            assert f.read(b, file_offset=start * a.itemsize) == b.nbytes
            xp.testing.assert_array_equal(a[start:end], b)

        with kvikio.RemoteFile.from_url(f"s3://{bucket_name}/{object_name}") as f:
            b = xp.zeros(shape=(end - start,), dtype=xp.int64)
            assert f.read(b, file_offset=start * a.itemsize) == b.nbytes
            xp.testing.assert_array_equal(a[start:end], b)
