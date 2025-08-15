# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import multiprocessing as mp
import time
from contextlib import contextmanager

import pytest
import utils

import kvikio
import kvikio.defaults

pytestmark = pytest.mark.skipif(
    not kvikio.is_remote_file_available(),
    reason=(
        "RemoteFile not available, please build KvikIO "
        "with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
    ),
)

# Notice, we import boto and moto after the `is_remote_file_available` check.
import boto3  # noqa: E402
import moto  # noqa: E402
import moto.server  # noqa: E402


@pytest.fixture(scope="session")
def endpoint_ip() -> str:
    return utils.localhost()


@pytest.fixture(scope="session")
def endpoint_port():
    return utils.find_free_port()


def start_s3_server(ip_address, port):
    server = moto.server.ThreadedMotoServer(ip_address=ip_address, port=port)
    server.start()
    time.sleep(600)
    print("ThreadedMotoServer shutting down because of timeout (10min)")


@pytest.fixture(scope="session")
def s3_base(endpoint_ip, endpoint_port):
    """Fixture to set up moto server in separate process"""
    with pytest.MonkeyPatch.context() as monkeypatch:
        # Use fake aws credentials
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "foobar_key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "foobar_secret")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        monkeypatch.setenv("AWS_ENDPOINT_URL", f"http://{endpoint_ip}:{endpoint_port}")

        p = mp.Process(target=start_s3_server, args=(endpoint_ip, endpoint_port))
        p.start()
        yield f"http://{endpoint_ip}:{endpoint_port}"
        p.kill()


@contextmanager
def s3_context(s3_base, bucket, files=None):
    if files is None:
        files = {}
    client = boto3.client("s3", endpoint_url=s3_base)
    client.create_bucket(Bucket=bucket, ACL="public-read-write")
    for f, data in files.items():
        client.put_object(Bucket=bucket, Key=f, Body=data)
    yield s3_base
    for f, data in files.items():
        try:
            client.delete_object(Bucket=bucket, Key=f)
        except Exception:
            pass


def test_read_access(s3_base):
    bucket_name = "bucket"
    object_name = "Data"
    data = b"file content"
    with s3_context(
        s3_base=s3_base, bucket=bucket_name, files={object_name: bytes(data)}
    ) as server_address:
        with kvikio.RemoteFile.open_s3_url(f"s3://{bucket_name}/{object_name}") as f:
            assert f.nbytes() == len(data)
            got = bytearray(len(data))
            assert f.read(got) == len(got)

        with kvikio.RemoteFile.open_s3(bucket_name, object_name) as f:
            assert f.nbytes() == len(data)
            got = bytearray(len(data))
            assert f.read(got) == len(got)

        with kvikio.RemoteFile.open_s3_url(
            f"{server_address}/{bucket_name}/{object_name}"
        ) as f:
            assert f.nbytes() == len(data)
            got = bytearray(len(data))
            assert f.read(got) == len(got)

        with pytest.raises(ValueError, match="Unsupported protocol"):
            kvikio.RemoteFile.open_s3_url(f"unknown://{bucket_name}/{object_name}")

        with pytest.raises(RuntimeError, match="URL returned error: 404"):
            kvikio.RemoteFile.open_s3("unknown-bucket", object_name)

        with pytest.raises(RuntimeError, match="URL returned error: 404"):
            kvikio.RemoteFile.open_s3(bucket_name, "unknown-file")


@pytest.mark.parametrize("size", [10, 100, 1000])
@pytest.mark.parametrize("nthreads", [1, 3])
@pytest.mark.parametrize("tasksize", [99, 999])
@pytest.mark.parametrize("buffer_size", [101, 1001])
def test_read(s3_base, xp, size, nthreads, tasksize, buffer_size):
    bucket_name = "test_read"
    object_name = "Aa1"
    a = xp.arange(size)
    with s3_context(
        s3_base=s3_base, bucket=bucket_name, files={object_name: bytes(a)}
    ) as server_address:
        with kvikio.defaults.set(
            {
                "num_threads": nthreads,
                "task_size": tasksize,
                "bounce_buffer_size": buffer_size,
            }
        ):
            with kvikio.RemoteFile.open_s3_url(
                f"{server_address}/{bucket_name}/{object_name}"
            ) as f:
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
        (42, int(2**20)),
    ],
)
def test_read_with_file_offset(s3_base, xp, start, end):
    bucket_name = "test_read_with_file_offset"
    object_name = "Aa1"
    a = xp.arange(end, dtype=xp.int64)
    with s3_context(
        s3_base=s3_base, bucket=bucket_name, files={object_name: bytes(a)}
    ) as server_address:
        url = f"{server_address}/{bucket_name}/{object_name}"
        with kvikio.RemoteFile.open_s3_url(url) as f:
            b = xp.zeros(shape=(end - start,), dtype=xp.int64)
            assert f.read(b, file_offset=start * a.itemsize) == b.nbytes
            xp.testing.assert_array_equal(a[start:end], b)
