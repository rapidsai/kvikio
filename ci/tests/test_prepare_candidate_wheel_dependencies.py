# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aligning candidate KvikIO wheel requirements to the frozen train."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPOSITORY_ROOT / "ci/prepare_candidate_wheel_dependencies.py"
_SOURCE_VERSION = (REPOSITORY_ROOT / "VERSION").read_text(encoding="utf-8").strip()
_SOURCE_MAJOR, _SOURCE_MINOR, *_ = _SOURCE_VERSION.split(".")
_SOURCE_WILDCARD = f"{_SOURCE_MAJOR}.{_SOURCE_MINOR}.*"
_SOURCE_REQUIREMENT = f"libkvikio=={_SOURCE_WILDCARD},>=0.0.0a0"
_SOURCE_WHEEL_REQUIREMENTS = {
    package: f"{package}=={_SOURCE_WILDCARD},>=0.0.0a0"
    for package in ("libkvikio-cu12", "libkvikio-cu13")
}
_FROZEN_VERSION = f"{_SOURCE_MAJOR}.{int(_SOURCE_MINOR) + 2}.0"
_INVALID_WILDCARD = f"{_SOURCE_MAJOR}.{int(_SOURCE_MINOR) + 2}.*"
_INVALID_PRERELEASE = f"{_FROZEN_VERSION}rc1"


def _write_fixture(root: Path) -> None:
    pyproject_path = root / "python/kvikio/pyproject.toml"
    pyproject_path.parent.mkdir(parents=True)
    pyproject_path.write_text(
        f'''[project]
dependencies = ["{_SOURCE_REQUIREMENT}"]

[tool.rapids-build-backend]
requires = ["{_SOURCE_REQUIREMENT}"]
''',
        encoding="utf-8",
    )
    (root / "dependencies.yaml").write_text(
        f"""build-use-libkvikio-wheel:
  common:
    - output_types: conda
      packages:
        - {_SOURCE_REQUIREMENT}
  specific:
    - output_types: [requirements, pyproject]
      packages:
        - {_SOURCE_WHEEL_REQUIREMENTS["libkvikio-cu12"]}
    - output_types: [requirements, pyproject]
      packages:
        - {_SOURCE_WHEEL_REQUIREMENTS["libkvikio-cu13"]}
depends_on_libkvikio:
  common:
    - output_types: conda
      packages:
        - {_SOURCE_REQUIREMENT}
  specific:
    - output_types: [requirements, pyproject]
      packages:
        - {_SOURCE_WHEEL_REQUIREMENTS["libkvikio-cu12"]}
    - output_types: [requirements, pyproject]
      packages:
        - {_SOURCE_WHEEL_REQUIREMENTS["libkvikio-cu13"]}
test_kvikio:
  common:
    - output_types: conda
      packages:
        - {_SOURCE_REQUIREMENT}
""",
        encoding="utf-8",
    )
    helper = root / "ci/prepare_candidate_wheel_dependencies.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HELPER, helper)


def _run_helper(
    root: Path,
    versions: dict[str, object] | None,
    *,
    candidate_mode: str = "true",
    write_fixture: bool = True,
) -> subprocess.CompletedProcess[str]:
    if write_fixture:
        _write_fixture(root)
    environment = os.environ.copy()
    environment.pop("RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS", None)
    environment["RAPIDS_RELEASE_CANDIDATE"] = candidate_mode
    if versions is not None:
        environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(versions)
    return subprocess.run(
        [sys.executable, str(root / "ci/prepare_candidate_wheel_dependencies.py")],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_candidate_rewrites_generated_runtime_and_build_wheel_requirements(
    tmp_path: Path,
) -> None:
    result = _run_helper(
        tmp_path,
        {"package_versions": {"wheel": _FROZEN_VERSION}, "unit_package_versions": {}},
    )

    assert result.returncode == 0, result.stderr
    pyproject = (tmp_path / "python/kvikio/pyproject.toml").read_text(encoding="utf-8")
    assert pyproject.count(f"libkvikio=={_FROZEN_VERSION}") == 2
    dependency_file = (tmp_path / "dependencies.yaml").read_text(encoding="utf-8")
    assert dependency_file.count(f"libkvikio-cu12=={_FROZEN_VERSION}") == 2
    assert dependency_file.count(f"libkvikio-cu13=={_FROZEN_VERSION}") == 2
    assert dependency_file.count(_SOURCE_REQUIREMENT) == 3
    assert dependency_file.count(f"=={_SOURCE_WILDCARD}") == 3


def test_owning_unit_override_updates_build_and_runtime_requirements(
    tmp_path: Path,
) -> None:
    result = _run_helper(
        tmp_path,
        {
            "package_versions": {"wheel": "26.14.0"},
            "unit_package_versions": {"wheel:kvikio": _FROZEN_VERSION},
        },
    )

    assert result.returncode == 0, result.stderr
    updated = (tmp_path / "python/kvikio/pyproject.toml").read_text(encoding="utf-8")
    assert updated.count(f"libkvikio=={_FROZEN_VERSION}") == 2
    dependency_file = (tmp_path / "dependencies.yaml").read_text(encoding="utf-8")
    assert dependency_file.count(f"libkvikio-cu12=={_FROZEN_VERSION}") == 2
    assert dependency_file.count(f"libkvikio-cu13=={_FROZEN_VERSION}") == 2


@pytest.mark.parametrize(
    "versions",
    [
        None,
        {"package_versions": {}, "unit_package_versions": {}},
        {"package_versions": {"wheel": _INVALID_WILDCARD}},
        {"package_versions": {"wheel": _INVALID_PRERELEASE}},
        {"package_versions": {"wheel": _FROZEN_VERSION}, "unit_package_versions": None},
    ],
)
def test_invalid_candidate_version_metadata_fails_without_changes(
    tmp_path: Path, versions: dict[str, object] | None
) -> None:
    _write_fixture(tmp_path)
    before_pyproject = (tmp_path / "python/kvikio/pyproject.toml").read_bytes()
    before_dependencies = (tmp_path / "dependencies.yaml").read_bytes()
    result = _run_helper(tmp_path, versions, write_fixture=False)

    assert result.returncode != 0
    assert (tmp_path / "python/kvikio/pyproject.toml").read_bytes() == before_pyproject
    assert (tmp_path / "dependencies.yaml").read_bytes() == before_dependencies


def test_ordinary_build_is_noop_without_candidate_metadata(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    before_pyproject = (tmp_path / "python/kvikio/pyproject.toml").read_bytes()
    before_dependencies = (tmp_path / "dependencies.yaml").read_bytes()
    result = _run_helper(tmp_path, None, candidate_mode="false", write_fixture=False)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "python/kvikio/pyproject.toml").read_bytes() == before_pyproject
    assert (tmp_path / "dependencies.yaml").read_bytes() == before_dependencies


def test_unexpected_generated_requirement_shape_fails_without_changes(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    dependency_path = tmp_path / "dependencies.yaml"
    dependency_path.write_text(
        dependency_path.read_text(encoding="utf-8").replace(
            _SOURCE_WHEEL_REQUIREMENTS["libkvikio-cu13"], "", 1
        ),
        encoding="utf-8",
    )
    before_pyproject = (tmp_path / "python/kvikio/pyproject.toml").read_bytes()
    before_dependencies = dependency_path.read_bytes()
    result = _run_helper(
        tmp_path,
        {"package_versions": {"wheel": _FROZEN_VERSION}},
        write_fixture=False,
    )

    assert result.returncode != 0
    assert (tmp_path / "python/kvikio/pyproject.toml").read_bytes() == before_pyproject
    assert dependency_path.read_bytes() == before_dependencies


@pytest.mark.parametrize("duplicate_in", ["project", "build"])
def test_misdistributed_pyproject_requirement_fails_without_changes(
    tmp_path: Path, duplicate_in: str
) -> None:
    _write_fixture(tmp_path)
    pyproject_path = tmp_path / "python/kvikio/pyproject.toml"
    original = pyproject_path.read_text(encoding="utf-8")
    if duplicate_in == "project":
        original = original.replace(
            f'dependencies = ["{_SOURCE_REQUIREMENT}"]',
            f'dependencies = ["{_SOURCE_REQUIREMENT}", "{_SOURCE_REQUIREMENT}"]',
        ).replace(f'requires = ["{_SOURCE_REQUIREMENT}"]', "requires = []")
    else:
        original = original.replace(
            f'dependencies = ["{_SOURCE_REQUIREMENT}"]', "dependencies = []"
        ).replace(
            f'requires = ["{_SOURCE_REQUIREMENT}"]',
            f'requires = ["{_SOURCE_REQUIREMENT}", "{_SOURCE_REQUIREMENT}"]',
        )
    pyproject_path.write_text(original, encoding="utf-8")
    before_pyproject = pyproject_path.read_bytes()
    before_dependencies = (tmp_path / "dependencies.yaml").read_bytes()
    result = _run_helper(
        tmp_path,
        {"package_versions": {"wheel": _FROZEN_VERSION}},
        write_fixture=False,
    )

    assert result.returncode != 0
    assert pyproject_path.read_bytes() == before_pyproject
    assert (tmp_path / "dependencies.yaml").read_bytes() == before_dependencies
