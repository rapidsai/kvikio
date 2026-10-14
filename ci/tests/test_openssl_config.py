# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test bundled OpenSSL initialization in fresh processes, without CUDA or conda."""

import http.server
import os
import ssl
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux wheel dependency"
)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    build = tmp_path_factory.mktemp("openssl-config-build")
    subprocess.run(
        ["cmake", "-S", str(ROOT / "ci/tests/openssl"), "-B", str(build)], check=True
    )
    subprocess.run(["cmake", "--build", str(build), "-j", "2"], check=True)
    return build / "openssl_config_probe"


def config(path, level):
    path.write_text(
        "openssl_conf = init\n[init]\nssl_conf = ssl\n"
        "[ssl]\nsystem_default = policy\n[policy]\n"
        f"CipherString = DEFAULT:@SECLEVEL={level}\n"
    )
    return path


def run(probe, first, second, override=None, endpoint=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENSSL_")}
    if override is not None:
        env["OPENSSL_CONF"] = str(override)
    command = [str(probe), str(first), str(second)]
    if endpoint is not None:
        command.append(endpoint)
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("use_first", [True, False])
def test_discovers_host_config(probe, tmp_path, use_first):
    first, second = tmp_path / "rhel.cnf", tmp_path / "debian.cnf"
    config(first if use_first else second, 3)
    result = run(probe, first, second)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3"


def test_override_wins(probe, tmp_path):
    host = config(tmp_path / "host.cnf", 3)
    override = config(tmp_path / "override.cnf", 4)
    result = run(probe, host, host, override)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4"


def test_empty_override_disables_host_config(probe, tmp_path):
    host = config(tmp_path / "host.cnf", 4)
    result = run(probe, host, host, "")
    baseline = run(probe, tmp_path / "absent1", tmp_path / "absent2")
    assert result.returncode == baseline.returncode == 0
    assert result.stdout == baseline.stdout
    assert result.stdout.strip() != "4"


def test_missing_explicit_config_fails(probe, tmp_path):
    host = config(tmp_path / "host.cnf", 3)
    result = run(probe, host, host, tmp_path / "missing.cnf")
    assert result.returncode != 0
    assert "Cannot load OpenSSL configuration" in result.stderr


@pytest.mark.parametrize(
    "contents",
    ["[unterminated", "openssl_conf = init\n[init]\nunknown_module = value\n"],
)
def test_invalid_selected_config_fails_without_fallback(probe, tmp_path, contents):
    broken = tmp_path / "broken.cnf"
    broken.write_text(contents)
    fallback = config(tmp_path / "fallback.cnf", 3)
    result = run(probe, broken, fallback)
    assert result.returncode != 0
    assert str(broken) in result.stderr


def test_system_config_includes_policy(probe, tmp_path):
    policy = config(tmp_path / "policy.cnf", 3)
    host = tmp_path / "openssl.cnf"
    host.write_text(f".include {policy}\n")
    result = run(probe, host, tmp_path / "missing")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3"


def test_directory_is_not_silently_skipped(probe, tmp_path):
    fallback = config(tmp_path / "fallback.cnf", 3)
    result = run(probe, tmp_path, fallback)
    assert result.returncode != 0
    assert "not a regular file" in result.stderr


def test_discovered_cipher_policy_rejects_handshake(probe, tmp_path):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-out",
            str(cert),
            "-keyout",
            str(key),
        ],
        check=True,
        capture_output=True,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_2
    context.set_ciphers("ECDHE-RSA-AES128-GCM-SHA256")
    context.load_cert_chain(cert, key)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), http.server.BaseHTTPRequestHandler
    )
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"127.0.0.1:{server.server_address[1]}"
        host = config(tmp_path / "host.cnf", 2)
        allowed = run(probe, host, tmp_path / "absent", endpoint=endpoint)
        assert allowed.returncode == 0, allowed.stderr
        host.write_text(
            host.read_text().replace(
                "DEFAULT:@SECLEVEL=2", "ECDHE-RSA-AES256-GCM-SHA384"
            )
        )
        rejected = run(probe, host, tmp_path / "absent", endpoint=endpoint)
        assert rejected.returncode != 0
        assert "TLS handshake rejected" in rejected.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
