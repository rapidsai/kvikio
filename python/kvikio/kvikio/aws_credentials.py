# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS credential sources for :meth:`kvikio.RemoteFile.open_s3` and related APIs.

These mirror the idea of a *credential* object (similar in spirit to Azure's
``DefaultAzureCredential`` and related types in the `Azure SDK for Python
<https://learn.microsoft.com/en-us/azure/developer/python/sdk/authorization/overview>`_),
but for AWS S3 access inside KvikIO's native code path.

KvikIO resolves credentials in C++ (with caching for IAM role credentials fetched via IMDSv2).
"""

from __future__ import annotations

from typing import Optional, Union

# Match ``kvikio::AwsCredentialKind`` (``remote_handle`` / C++).
_CRED_DEFAULT: int = 0
_CRED_ENVIRONMENT: int = 1
_CRED_STATIC: int = 2
_CRED_IAM_ROLE: int = 3
_CRED_LEGACY: int = 4


class AwsDefaultCredential:
    """Use environment variables if set, otherwise an IAM role via the metadata service (IMDSv2).

    If both ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` are set and non-empty,
    those are used. Otherwise KvikIO fetches temporary credentials for the IAM role
    attached to the compute environment (e.g. EC2 instance, Lambda, ECS task) from the
    metadata endpoint (see `IAM roles for EC2
    <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html>`_
    and related services).

    Attributes
    ----------
    imds_endpoint : str or None
        Optional metadata service base URL (e.g. ``http://127.0.0.1:1234``) for tests.
        If ``None``, uses ``AWS_EC2_METADATA_SERVICE_ENDPOINT`` when set, else the default
        link-local address.
    """

    __slots__ = ("imds_endpoint",)

    def __init__(self, imds_endpoint: Optional[str] = None) -> None:
        self.imds_endpoint = imds_endpoint

    def _kvikio_kind(self) -> int:
        return _CRED_DEFAULT

    def _kvikio_imds_endpoint(self) -> Optional[str]:
        return self.imds_endpoint


class AwsEnvironmentCredential:
    """Read credentials only from the environment (no IAM-role / metadata fallback).

    Uses ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and optionally
    ``AWS_SESSION_TOKEN`` (required for temporary ``ASIA`` access keys).
    """

    __slots__ = ()

    def _kvikio_kind(self) -> int:
        return _CRED_ENVIRONMENT

    def _kvikio_imds_endpoint(self) -> Optional[str]:
        return None


class AwsLegacyCredential:
    """Same credential resolution as the pre-``credential=`` S3 API (optional args + env).

    Each field may be ``None``; missing values are taken from ``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, and ``AWS_SESSION_TOKEN`` when applicable. This matches the
    behavior of the deprecated ``aws_access_key_id=`` / ``aws_secret_access_key=`` /
    ``aws_session_token=`` keyword arguments on :meth:`kvikio.RemoteFile.open_s3`.
    """

    __slots__ = ("aws_access_key_id", "aws_secret_access_key", "aws_session_token")

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ) -> None:
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token

    def _kvikio_kind(self) -> int:
        return _CRED_LEGACY

    def _kvikio_imds_endpoint(self) -> Optional[str]:
        return None


class AwsStaticCredential:
    """Fixed access key, secret key, and optional session token (no environment lookup).

    Attributes
    ----------
    access_key_id : str
    secret_access_key : str
    session_token : str or None
        Required when ``access_key_id`` begins with ``ASIA`` (temporary credentials).
    """

    __slots__ = ("access_key_id", "secret_access_key", "session_token")

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        session_token: Optional[str] = None,
    ) -> None:
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token

    def _kvikio_kind(self) -> int:
        return _CRED_STATIC

    def _kvikio_imds_endpoint(self) -> Optional[str]:
        return None


class AwsIamRoleCredential:
    """IAM role credentials from the compute metadata service (IMDSv2) only.

    Ignores static ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` environment
    variables and always uses the role credentials exposed at the metadata endpoint
    (used on EC2, Lambda, ECS, and other AWS compute).

    Attributes
    ----------
    imds_endpoint : str or None
        Optional metadata service base URL; same semantics as
        :attr:`AwsDefaultCredential.imds_endpoint`.
    """

    __slots__ = ("imds_endpoint",)

    def __init__(self, imds_endpoint: Optional[str] = None) -> None:
        self.imds_endpoint = imds_endpoint

    def _kvikio_kind(self) -> int:
        return _CRED_IAM_ROLE

    def _kvikio_imds_endpoint(self) -> Optional[str]:
        return self.imds_endpoint


AwsCredential = Union[
    AwsDefaultCredential,
    AwsEnvironmentCredential,
    AwsLegacyCredential,
    AwsStaticCredential,
    AwsIamRoleCredential,
]

__all__ = [
    "AwsCredential",
    "AwsDefaultCredential",
    "AwsIamRoleCredential",
    "AwsEnvironmentCredential",
    "AwsLegacyCredential",
    "AwsStaticCredential",
]
