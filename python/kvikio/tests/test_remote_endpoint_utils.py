# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import kvikio

pytestmark = pytest.mark.skipif(
    not kvikio.is_remote_file_available(),
    reason=(
        "RemoteFile not available, please build KvikIO "
        "with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
    ),
)


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")


def test_infer_remote_endpoint_type(aws_env):
    assert (
        kvikio.infer_remote_endpoint_type("s3://bucket-name/object-key-name")
        == kvikio.RemoteEndpointType.S3
    )
    assert (
        kvikio.infer_remote_endpoint_type("https://host:1234/webhdfs/v1/data.bin")
        == kvikio.RemoteEndpointType.WEBHDFS
    )
    assert (
        kvikio.infer_remote_endpoint_type("https://example.com/path/file.bin")
        == kvikio.RemoteEndpointType.HTTP
    )
    assert (
        kvikio.infer_remote_endpoint_type(
            "https://bucket-name.s3.region-code.amazonaws.com/"
            "object-key-name?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=sig&"
            "X-Amz-Credential=cred&X-Amz-SignedHeaders=host"
        )
        == kvikio.RemoteEndpointType.S3_PRESIGNED_URL
    )


def test_infer_remote_endpoint_type_invalid_url():
    with pytest.raises(RuntimeError, match="Bad scheme"):
        kvikio.infer_remote_endpoint_type("example.com/path")

    with pytest.raises(RuntimeError, match="Unsupported endpoint URL"):
        kvikio.infer_remote_endpoint_type("unsupported://example.com/path")
