# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Align KvikIO wheel metadata with the version frozen in a candidate train."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+")
_PYPROJECT_REQUIREMENT = re.compile(r"libkvikio==\d+\.\d+\.\*,>=0\.0\.0a0")
_WHEEL_REQUIREMENT = re.compile(
    r"(?P<package>libkvikio-cu12|libkvikio-cu13)==\d+\.\d+\.\*,>=0\.0\.0a0"
)
_WHEEL_PACKAGES = {"libkvikio-cu12", "libkvikio-cu13"}


def _candidate_version(raw_versions: str) -> str:
    """Return the frozen wheel version, preferring the unit override."""
    try:
        versions: Any = json.loads(raw_versions)
    except json.JSONDecodeError as exc:
        raise ValueError("candidate unit-version metadata is not valid JSON") from exc

    if not isinstance(versions, dict):
        raise ValueError("candidate unit-version metadata must be an object")

    unit_versions = versions.get("unit_package_versions", {})
    family_versions = versions.get("package_versions", {})
    if not isinstance(unit_versions, dict) or not isinstance(family_versions, dict):
        raise ValueError("candidate version maps must be objects")

    # The wheel:kvikio release unit owns both the kvikio and libkvikio wheel
    # artifacts, so its override also governs this build dependency.
    version = unit_versions.get("wheel:kvikio", family_versions.get("wheel"))
    if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError("frozen wheel:kvikio version is missing or invalid")
    return version


def _rewrite_pyproject(path: Path, version: str) -> str:
    """Return generated pyproject text with both wheel constraints updated."""
    import tomllib

    original = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(original)
    requirement_lists = (
        parsed.get("project", {}).get("dependencies", []),
        parsed.get("tool", {}).get("rapids-build-backend", {}).get("requires", []),
    )
    selected = [
        [
            requirement
            for requirement in requirements
            if isinstance(requirement, str) and requirement.startswith("libkvikio==")
        ]
        for requirements in requirement_lists
    ]
    if any(
        len(requirements) != 1 or not _PYPROJECT_REQUIREMENT.fullmatch(requirements[0])
        for requirements in selected
    ):
        raise ValueError(
            "expected one generated wildcard libkvikio requirement in "
            "project.dependencies and tool.rapids-build-backend.requires"
        )

    updated, count = _PYPROJECT_REQUIREMENT.subn(f"libkvikio=={version}", original)
    if count != 2:
        raise ValueError("expected exactly two generated libkvikio requirements")
    return updated


def _rewrite_dependency_file(path: Path, version: str) -> str:
    """Update only generated CUDA wheel requirements, preserving Conda pins."""
    original = path.read_text(encoding="utf-8")
    matches = list(_WHEEL_REQUIREMENT.finditer(original))
    packages = [match.group("package") for match in matches]
    if (
        len(matches) != 4
        or packages.count("libkvikio-cu12") != 2
        or packages.count("libkvikio-cu13") != 2
    ):
        raise ValueError(
            "expected two generated wheel requirements for each CUDA family"
        )

    return _WHEEL_REQUIREMENT.sub(
        lambda match: f"{match.group('package')}=={version},>=0.0.0a0",
        original,
    )


def main() -> int:
    """Update candidate metadata; leave ordinary production builds untouched."""
    candidate_mode = os.environ.get("RAPIDS_RELEASE_CANDIDATE", "false").lower()
    if candidate_mode not in {"true", "false"}:
        print("RAPIDS_RELEASE_CANDIDATE must be true or false", file=sys.stderr)
        return 1
    if candidate_mode == "false":
        return 0

    try:
        raw_versions = os.environ.get("RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS")
        if raw_versions is None:
            raise ValueError("candidate unit-version metadata is required")
        version = _candidate_version(raw_versions)
        pyproject_path = Path("python/kvikio/pyproject.toml")
        dependency_file_path = Path("dependencies.yaml")
        pyproject = _rewrite_pyproject(pyproject_path, version)
        dependency_file = _rewrite_dependency_file(dependency_file_path, version)
        pyproject_path.write_text(pyproject, encoding="utf-8")
        dependency_file_path.write_text(dependency_file, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(
            f"Unable to prepare frozen KvikIO candidate requirements: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Applied frozen candidate libkvikio=={version} to KvikIO wheel metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
