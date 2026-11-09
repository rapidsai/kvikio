# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real recipe script inside Rattler's isolated build shell."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("policy", ["false", "true"])
def test_isolated_recipe_respects_candidate_cache_policy(tmp_path, policy):
    rattler = shutil.which("rattler-build")
    if rattler is None:
        pytest.skip("rattler-build is required for the isolated-shell integration")
    original = yaml.safe_load(
        (ROOT / "conda/recipes/libkvikio/recipe.yaml").read_text()
    )
    script = original["outputs"][0]["build"]["script"]
    assert script["env"]["RAPIDS_RELEASE_CANDIDATE_SCCACHE"] == (
        '${{ env.get("RAPIDS_RELEASE_CANDIDATE_SCCACHE", default="true") }}'
    )
    source = tmp_path / "source"
    source.mkdir()
    build = source / "build.sh"
    build.write_text(
        "#!/bin/bash\nset -eu\n"
        'if [[ -n "${CMAKE_CXX_COMPILER_LAUNCHER:-}" ]]; then\n'
        '  "$CMAKE_CXX_COMPILER_LAUNCHER" /usr/bin/true\n'
        "fi\n"
        'printf "direct-compiler\\n" > "$PREFIX/cache-policy-result"\n'
    )
    build.chmod(0o755)
    cache = tmp_path / "sccache"
    cache.write_text(
        "#!/bin/sh\n"
        'echo "sccache: error: SCCACHE_DIST_AUTH: missing field token" >&2\n'
        "exit 42\n"
    )
    cache.chmod(0o755)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "context": {"cuda_major": "13"},
                "package": {"name": "candidate-cache-fixture", "version": "0"},
                "source": {"path": str(source)},
                "build": {"script": script},
            },
            sort_keys=False,
        )
    )
    environment = {
        **os.environ,
        "RAPIDS_RELEASE_CANDIDATE_SCCACHE": policy,
        "CMAKE_C_COMPILER_LAUNCHER": str(cache),
        "CMAKE_CXX_COMPILER_LAUNCHER": str(cache),
        "CMAKE_CUDA_COMPILER_LAUNCHER": str(cache),
        "CMAKE_GENERATOR": "Ninja",
        "CFLAGS": "",
        "CXXFLAGS": "",
        "RAPIDS_CONDA_ARCH": "x86_64",
        "CONDA_OVERRIDE_CUDA": "13.4",
    }
    environment.pop("SCCACHE_DIST_AUTH_TOKEN", None)
    result = subprocess.run(  # noqa: S603 -- fixed CLI and repository-owned fixture recipe
        [
            rattler,
            "build",
            "--recipe",
            str(recipe),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-test",
            "--color",
            "never",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    log = result.stdout + result.stderr
    if policy == "false":
        assert result.returncode == 0, log
        assert list((tmp_path / "output").rglob("*.conda"))
        assert "missing field token" not in log
    else:
        assert result.returncode != 0
        assert "missing field token" in log
