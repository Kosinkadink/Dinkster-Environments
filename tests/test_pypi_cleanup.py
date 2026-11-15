from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pypi_cleanup


def releases(*versions: str) -> dict[str, list[dict[str, str]]]:
    return {
        version: [{"packagetype": "bdist_wheel", "filename": f"package-{version}.whl"}]
        for version in versions
    }


def inventory() -> dict:
    sources = []
    coverage = {}
    for kind in pypi_cleanup.REQUIRED_SOURCE_KINDS:
        identifier = kind + ":main"
        sources.append(
            {
                "id": identifier,
                "kind": kind,
                "location": "https://example.invalid/" + identifier,
                "sha256": hashlib.sha256(identifier.encode()).hexdigest(),
            }
        )
        coverage[kind] = [identifier]
    return {
        "format": "dinkster.pypi-cleanup-inventory/1",
        "sources": sources,
        "coverage": coverage,
        "pins": [],
    }


def test_previous_stable_boundary_deletes_older_development_wheels() -> None:
    result = pypi_cleanup.plan_project(
        "dinkster",
        releases("1.0.0.dev1", "1.0.0", "1.1.0.dev1", "1.1.0", "1.2.0.dev1", "1.2.0"),
        [],
    )
    assert result["previousStable"] == "1.1.0"
    assert result["candidates"] == [
        {"version": "1.0.0.dev1", "protectedBy": []},
        {"version": "1.1.0.dev1", "protectedBy": []},
    ]


def test_projects_use_independent_stable_version_lines() -> None:
    snapshot = inventory()
    parsed = {
        "sources": {source["id"]: source for source in snapshot["sources"]},
        "protected": {project: [] for project in pypi_cleanup.PROJECTS},
    }
    data = {
        project: releases("1.0.0.dev1", "1.0.0") for project in pypi_cleanup.PROJECTS
    }
    data["dinkster"] = releases("1.0.0.dev1", "1.0.0", "1.1.0")
    data["dinkster-api"] = releases("7.0.0.dev1", "7.0.0", "7.1.0.dev1", "7.1.0")
    result = pypi_cleanup.plan(parsed, data)
    assert result["projects"]["dinkster"]["candidates"] == [
        {"version": "1.0.0.dev1", "protectedBy": []}
    ]
    assert result["projects"]["dinkster-api"]["candidates"] == [
        {"version": "7.0.0.dev1", "protectedBy": []}
    ]


def test_old_pinned_development_wheel_is_never_a_candidate() -> None:
    result = pypi_cleanup.plan_project(
        "dinkster",
        releases("1.0.0.dev1", "1.0.0.dev2", "1.0.0", "1.1.0"),
        [{"version": "1.0.0.dev1", "source": "retained-engine-manifest:one"}],
    )
    assert result["candidates"] == [{"version": "1.0.0.dev2", "protectedBy": []}]


def test_incomplete_protection_data_refuses_to_plan(tmp_path: Path) -> None:
    document = inventory()
    document["coverage"].pop("dependency-lock")
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="coverage is incomplete: dependency-lock"):
        pypi_cleanup.read_inventory(path)
