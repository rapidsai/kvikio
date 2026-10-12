# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security and real-operation contracts for the nVault workflow test."""

import base64
import gzip
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "credential_test", Path(__file__).parents[1] / "credential_test.py"
)
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


@pytest.fixture
def environment():
    return {
        "NVAULT_JOB_JWT": "dummy-job-jwt",
        "NVAULT_AUTH_MOUNT": "jwt/github-actions/literate-octo-apple",
        "NVAULT_ROLE": "rapids-pulse-credential-test",
        "NVAULT_SSA_MOUNT": CLIENT.SSA_MOUNTS[0],
        "NVAULT_EXPECTED_SERVICE_ID": "dummy-owner",
        "NSPECT_SSA_SCOPE": "public.api:read",
        "NSPECT_ID": "NSPECT-1RDT-NZVP",
    }


def fake_requests(environment, calls, broad=False):
    def request(url, headers, data=None, *, phase):
        calls.append((url, headers.copy(), data, phase))
        if url.endswith("/login"):
            return {"auth": {"client_token": "dummy-vault-token"}}
        if url.endswith("/capabilities-self"):
            paths = json.loads(data)["paths"]
            result = {path: ["deny"] for path in paths}
            result[paths[0]] = ["read"]
            if broad:
                result[paths[1]] = ["read", "update"]
            return {"data": result}
        if url.endswith("/issue/creds"):
            return {
                "data": {
                    "client_id": "dummy-client",
                    "secret": "dummy-secret",
                    "service_id": environment["NVAULT_EXPECTED_SERVICE_ID"],
                }
            }
        if url == CLIENT.OAUTH_ENDPOINT:
            return {"access_token": "dummy-oauth-token"}
        if url.endswith("/programVersions"):
            return {"programVersions": []}
        raise AssertionError("Unexpected request")

    return request


def test_reads_api_using_fetched_credentials_without_logging(
    environment, monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(CLIENT, "request_json", fake_requests(environment, calls))
    CLIENT.run(environment)
    output = capsys.readouterr()
    assert "PASS: nSpect program-version read" in output.out
    for sensitive in (
        "dummy-vault-token",
        "dummy-client",
        "dummy-secret",
        "dummy-oauth-token",
    ):
        assert sensitive not in output.out + output.err
    oauth = next(call for call in calls if call[0] == CLIENT.OAUTH_ENDPOINT)
    expected = base64.b64encode(b"dummy-client:dummy-secret").decode()
    assert oauth[1]["Authorization"] == f"Basic {expected}"
    assert calls[-1][1] == {"Authorization": "Bearer dummy-oauth-token"}
    assert "grant_type=client_credentials" in oauth[2].decode()


def test_excess_permissions_stop_before_credential_read(environment, monkeypatch):
    calls = []
    monkeypatch.setattr(
        CLIENT, "request_json", fake_requests(environment, calls, broad=True)
    )
    with pytest.raises(RuntimeError, match="unexpected SSA"):
        CLIENT.run(environment)
    assert not any(call[0].endswith("/issue/creds") for call in calls)


@pytest.mark.parametrize("status", [400, 403])
def test_real_auth_denial_is_accepted(environment, monkeypatch, status):
    def denied(*args, **kwargs):
        raise CLIENT.RequestFailure("login", status, "claim-validation")

    monkeypatch.setattr(CLIENT, "request_json", denied)
    CLIENT.expect_login_denied(environment, "dummy-jwt", "wrong audience")


@pytest.mark.parametrize("status", [404, 500])
def test_other_http_failures_are_not_counted_as_denial(
    environment, monkeypatch, status
):
    def denied(*args, **kwargs):
        raise CLIENT.RequestFailure("login", status)

    monkeypatch.setattr(CLIENT, "request_json", denied)
    with pytest.raises(CLIENT.RequestFailure):
        CLIENT.expect_login_denied(environment, "dummy-jwt", "wrong audience")


def test_network_failure_is_not_a_successful_denial(environment, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(CLIENT, "request_json", failed)
    with pytest.raises(RuntimeError, match="network"):
        CLIENT.expect_login_denied(environment, "dummy-jwt", "wrong branch")


@pytest.mark.parametrize("category", ["unclassified", "network-policy"])
def test_http_403_without_claim_validation_is_not_a_pass(
    environment, monkeypatch, category
):
    def denied(*args, **kwargs):
        raise CLIENT.RequestFailure("login", 403, category)

    monkeypatch.setattr(CLIENT, "request_json", denied)
    with pytest.raises(CLIENT.RequestFailure):
        CLIENT.expect_login_denied(environment, "dummy-jwt", "wrong branch")


def test_oidc_request_uses_explicit_audience_and_no_token_output(monkeypatch):
    calls = []

    def issue(url, headers, **kwargs):
        calls.append((url, headers))
        return {"value": "header.payload.signature"}

    monkeypatch.setattr(CLIENT, "request_json", issue)
    result = CLIENT.github_jwt(
        {
            "ACTIONS_ID_TOKEN_REQUEST_URL": "https://pipelines.actions.githubusercontent.com/idtoken?api-version=2",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "dummy-request-token",
        },
        "https://prod.vault.nvidia.com:443",
    )
    assert result == "header.payload.signature"
    assert "audience=https%3A%2F%2Fprod.vault.nvidia.com%3A443" in calls[0][0]
    assert "dummy-request-token" not in calls[0][0]


def test_refuses_redirect_and_external_oidc_destination():
    with pytest.raises(RuntimeError, match="redirects"):
        CLIENT.NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://evil.example"
        )
    with pytest.raises(ValueError, match="approved"):
        CLIENT.github_jwt(
            {
                "ACTIONS_ID_TOKEN_REQUEST_URL": "https://evil.example",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "dummy",
            },
            "audience",
        )


def test_http_errors_do_not_echo_response_body(monkeypatch):
    class Opener:
        def open(self, *args, **kwargs):
            raise urllib.error.HTTPError(
                "https://example.test", 403, "sensitive-detail", {}, None
            )

    monkeypatch.setattr(CLIENT.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(CLIENT.RequestFailure) as failure:
        CLIENT.request_json("https://example.test", {}, phase="read")
    assert "sensitive-detail" not in str(failure.value)


@pytest.mark.parametrize(
    ("status", "body", "encoding", "category"),
    [
        (
            400,
            b'{"errors":["rejected assertion"]}',
            "",
            "vault-request-rejected",
        ),
        (400, b'{"errors":["E10200"]}', "", "network-policy"),
        (403, b'{"errors":["permission denied"]}', "", "unclassified"),
        (400, b"<html>gateway failure</html>", "", "unclassified"),
        (
            400,
            gzip.compress(b'{"errors":["does not match bound claims"]}'),
            "gzip",
            "claim-validation",
        ),
    ],
)
def test_denial_format_distinguishes_backend_and_gateway(
    monkeypatch, status, body, encoding, category
):
    class Opener:
        def open(self, *args, **kwargs):
            raise urllib.error.HTTPError(
                "https://example.test",
                status,
                "",
                {"Content-Encoding": encoding},
                io.BytesIO(body),
            )

    monkeypatch.setattr(CLIENT.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(CLIENT.RequestFailure) as failure:
        CLIENT.request_json("https://example.test", {}, phase="login")
    assert failure.value.category == category
