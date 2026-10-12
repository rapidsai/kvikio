# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitHub OIDC -> nVault SSA -> read-only nSpect API credential test.

Adapted from pulse-scanning's October 2026 nvault_ssa_canary.py. Credentials
remain within this process; only selected identity metadata and pass/fail
messages reach the workflow log.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

VAULT_ADDR = "https://prod.vault.nvidia.com"
VAULT_NAMESPACE = "swgpu-rapids"
OAUTH_ENDPOINT = (
    "https://4ubglassowmtsi7ogqwarmut7msn1q5ynts62fwnr1i.ssa.nvidia.com/token"
)
API_BASE = "https://nspect.nvidia.com/pm/api/v1.0"
SSA_MOUNTS = (
    "services/rapids-pulse/ssa/prod",
    "services/librtcx-scans/ssa/prod",
    "services/libcuvs-static/ssa/prod",
    "services/libcudf-static/ssa/prod",
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("HTTP redirects are refused")


class RequestFailure(RuntimeError):
    """Sanitized failure with an HTTP status for actual denial checks."""

    def __init__(self, phase, status, category="unclassified"):
        super().__init__(f"{phase} failed (HTTP {status}; {category})")
        self.status = status
        self.category = category


def request_json(url, headers, data=None, *, phase):
    request = urllib.request.Request(url, headers=headers, data=data)
    try:
        opener = urllib.request.build_opener(NoRedirect())
        # Callers constrain URLs to approved HTTPS origins.
        with opener.open(request, timeout=30) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Classify only known markers; never echo response bodies or claims.
        raw = exc.read(65536)
        if exc.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        detail = raw.decode("utf-8", errors="replace").lower()
        category = "unclassified"
        if any(code in detail for code in ("e10001", "e10111", "e10200")):
            category = "network-policy"
        elif (
            "bound audience" in detail
            or "bound claim" in detail
            or "audience claim does not match any expected audience" in detail
        ):
            category = "claim-validation"
        elif exc.code == 400:
            # Some nVault frontends replace validation text with error codes.
            # Require its structured error response, not an HTML gateway page.
            try:
                rejected = json.loads(raw)
            except ValueError:
                rejected = None
            if (
                isinstance(rejected, dict)
                and isinstance(rejected.get("errors"), list)
                and rejected["errors"]
                and all(isinstance(item, str) for item in rejected["errors"])
            ):
                category = "vault-request-rejected"
        raise RequestFailure(phase, exc.code, category) from None
    except OSError as exc:
        # Report the exception class only; URL, headers, and response bodies may
        # contain credentials or sensitive service details.
        raise RuntimeError(f"{phase} failed (network: {type(exc).__name__})") from None
    except ValueError:
        raise RuntimeError(f"{phase} failed (invalid JSON response)") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"{phase} returned an invalid response")
    return payload


def required(environ, key):
    value = environ.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def github_jwt(environ, audience):
    """Request a real job JWT without exporting it as a workflow output."""
    url = required(environ, "ACTIONS_ID_TOKEN_REQUEST_URL")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".actions.githubusercontent.com")
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("GitHub OIDC endpoint is not an approved HTTPS origin")
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parsed.query) if k != "audience"]
    query.append(("audience", audience))
    url = urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))
    payload = request_json(
        url,
        {
            "Authorization": "Bearer "
            + required(environ, "ACTIONS_ID_TOKEN_REQUEST_TOKEN")
        },
        phase="GitHub OIDC request",
    )
    jwt = payload.get("value")
    if not isinstance(jwt, str) or len(jwt.split(".")) != 3:
        raise RuntimeError("GitHub OIDC endpoint returned no valid JWT")
    return jwt


def print_identity(jwt):
    """Print selected non-secret, unverified identity claims only."""
    try:
        part = jwt.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except (ValueError, IndexError):
        raise ValueError("GitHub JWT payload is invalid") from None
    if not isinstance(claims, dict):
        raise ValueError("GitHub JWT claims are invalid")
    names = (
        "iss",
        "aud",
        "repository_id",
        "repository_owner_id",
        "workflow_ref",
        "sha",
        "ref",
        "event_name",
        "environment",
    )
    print("Unverified identity metadata; nVault must validate the signature:")
    print(json.dumps({name: claims.get(name) for name in names}, sort_keys=True))


def expect_login_denied(environ, jwt, reason):
    """Count only an actual login authorization response as denial."""
    auth_mount = required(environ, "NVAULT_AUTH_MOUNT")
    role = required(environ, "NVAULT_ROLE")
    try:
        request_json(
            f"{VAULT_ADDR}/v1/auth/{auth_mount}/login",
            {
                "X-Vault-Namespace": VAULT_NAMESPACE,
                "Content-Type": "application/json",
            },
            json.dumps({"role": role, "jwt": jwt}).encode(),
            phase="Negative Vault JWT login",
        )
    except RequestFailure as exc:
        if (exc.status in (400, 403) and exc.category == "claim-validation") or (
            exc.status == 400 and exc.category == "vault-request-rejected"
        ):
            print(f"PASS: {reason} denied by nVault (HTTP {exc.status})")
            return
        raise
    raise RuntimeError(f"nVault unexpectedly accepted {reason}")


def run(environ: Mapping[str, str]):
    # Validate every job input before sending the workload assertion.
    jwt = required(environ, "NVAULT_JOB_JWT")
    auth_mount = required(environ, "NVAULT_AUTH_MOUNT")
    role = required(environ, "NVAULT_ROLE")
    mount = required(environ, "NVAULT_SSA_MOUNT")
    service_id = required(environ, "NVAULT_EXPECTED_SERVICE_ID")
    scope = required(environ, "NSPECT_SSA_SCOPE")
    nspect_id = required(environ, "NSPECT_ID")
    if mount not in SSA_MOUNTS:
        raise ValueError(
            "NVAULT_SSA_MOUNT must select one of the four documented mounts"
        )
    if any(part in ("", ".", "..") for part in auth_mount.split("/")) or any(
        not (c.isascii() and (c.isalnum() or c in "/_-")) for c in auth_mount
    ):
        raise ValueError(
            "NVAULT_AUTH_MOUNT must be a relative auth mount, without auth/"
        )
    if auth_mount.startswith("auth/"):
        raise ValueError("NVAULT_AUTH_MOUNT must omit the auth/ prefix")

    headers = {
        "X-Vault-Namespace": VAULT_NAMESPACE,
        "Content-Type": "application/json",
    }
    login = request_json(
        f"{VAULT_ADDR}/v1/auth/{auth_mount}/login",
        headers,
        json.dumps({"role": role, "jwt": jwt}).encode(),
        phase="Vault JWT login",
    )
    auth = login.get("auth")
    token = auth.get("client_token") if isinstance(auth, dict) else None
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("Vault JWT login returned no token")
    headers["X-Vault-Token"] = token
    print("PASS: GitHub workload JWT login")

    paths = [
        f"{mount}/issue/creds",
        f"{mount}/config",
        f"{mount}/secret/rotate",
        f"{mount}/secret/commit",
        f"{mount}/config/reset",
    ]
    paths.extend(f"{other}/issue/creds" for other in SSA_MOUNTS if other != mount)
    capabilities = request_json(
        f"{VAULT_ADDR}/v1/sys/capabilities-self",
        headers,
        json.dumps({"paths": paths}).encode(),
        phase="Vault capability check",
    )
    capability_map = capabilities.get("data", capabilities)
    if not isinstance(capability_map, dict):
        raise RuntimeError("Vault capability check returned invalid data")
    if set(capability_map.get(paths[0], [])) != {"read"}:
        raise RuntimeError(
            "Runtime role must grant exactly read on the selected issue/creds"
        )
    for path in paths[1:]:
        if capability_map.get(path) != ["deny"]:
            raise RuntimeError(
                "Runtime role grants unexpected SSA administrative or other-client access"
            )
    print("PASS: SSA read policy isolation")

    payload = request_json(
        f"{VAULT_ADDR}/v1/{mount}/issue/creds",
        headers,
        phase="SSA credential read",
    )
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("service_id") != service_id:
        raise RuntimeError("SSA response does not match the expected owning Service ID")
    cid, secret = data.get("client_id"), data.get("secret")
    if not all(isinstance(value, str) and value.strip() for value in (cid, secret)):
        raise RuntimeError("SSA response is missing client_id or secret")
    print("PASS: expected SSA client service")

    basic = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    exchange = request_json(
        OAUTH_ENDPOINT,
        {
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        urllib.parse.urlencode(
            {"grant_type": "client_credentials", "scope": scope}
        ).encode(),
        phase="Starfleet OAuth exchange",
    )
    access_token = exchange.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise RuntimeError("Starfleet OAuth response is missing access_token")
    print("PASS: nSpect-scoped Starfleet exchange")
    request_json(
        f"{API_BASE}/public/programs/{urllib.parse.quote(nspect_id, safe='')}/programVersions",
        {"Authorization": f"Bearer {access_token}"},
        phase="nSpect program-version read",
    )
    print("PASS: nSpect program-version read")


def main():
    try:
        if sys.argv[1:] not in ([], ["--expect-login-denied"]):
            raise ValueError("Only --expect-login-denied is supported")
        environ = dict(os.environ)
        audience = required(environ, "NVAULT_AUDIENCE")
        jwt = github_jwt(environ, audience)
        print_identity(jwt)
        if sys.argv[1:] == ["--expect-login-denied"]:
            expect_login_denied(environ, jwt, "wrong branch/workflow identity")
        else:
            environ["NVAULT_JOB_JWT"] = jwt
            run(environ)
            wrong_jwt = github_jwt(environ, audience + "/wrong-audience")
            expect_login_denied(environ, wrong_jwt, "wrong audience")
    except (ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
