from __future__ import annotations

import json

import pytest

import pypi_inventory


def report() -> dict:
    entries = [
        {"path": "engine/current", "scope": "engine", "tag": "v1.1.0", "releaseId": 1},
        {
            "path": "base/linux/current",
            "scope": "base",
            "tag": "base-linux",
            "releaseId": 2,
        },
        {
            "path": "tools/current",
            "scope": "tools",
            "tag": "tools-current",
            "releaseId": 3,
        },
    ]
    return {
        "unresolvedConsumerPins": [],
        "unclassifiedDirectories": [],
        "publications": entries,
        "keep": [entry["path"] for entry in entries],
    }


def test_build_inventory_collects_retained_and_current_pins(monkeypatch) -> None:
    engine = {
        "format": "dinkster.engine/1",
        "workspaceWheels": ["dinkster-1.0.0.dev1-py3-none-any.whl"],
        "dependencies": {
            "linux": {
                "wheels": [
                    {"name": "dinkster-engine", "version": "1.0.0.dev2"},
                    {"name": "dinkster-inference", "version": "1.0.0.dev3"},
                    {"name": "dinkster-token-verifier", "version": "1.0.0.dev4"},
                ]
            }
        },
    }
    assets = {
        (pypi_inventory.ENGINE_REPOSITORY, 1, "release-manifest.json"): engine,
        ("Kosinkadink/Dinkster-Environments", 2, "manifest.json"): {
            "format": "dinkster.base/1",
            "wheels": [{"name": "dinkster-api", "version": "1.0.0.dev5"}],
        },
        (pypi_inventory.ENGINE_REPOSITORY, 3, "tools-manifest.json"): {
            "format": "dinkster.tools/1"
        },
    }
    files = {
        (pypi_inventory.ENGINE_REPOSITORY, "scripts/release_sources.json"): {
            "version": "1.0.0.dev6"
        },
        (pypi_inventory.DESKTOP_REPOSITORY, "scripts/frontend-release.json"): {
            "version": "1.0.0.dev6"
        },
        (pypi_inventory.ENGINE_REPOSITORY, "pyproject.toml"): {
            "text": '[project]\ndependencies = ["dinkster-supervisor==1.0.0.dev7"]\n'
        },
        ("Kosinkadink/dinkster-nodes", "pyproject.toml"): {"text": "[project]\n"},
        (
            pypi_inventory.DESKTOP_REPOSITORY,
            "packages/desktop/src/backend-release.json",
        ): {"version": "1.0.0.dev1"},
    }
    monkeypatch.setattr(
        pypi_inventory,
        "release_asset",
        lambda repository, release_id, name: json.dumps(
            assets[repository, release_id, name]
        ).encode(),
    )
    monkeypatch.setattr(
        pypi_inventory,
        "github_file",
        lambda repository, path: (
            files[repository, path]
            .get("text", json.dumps(files[repository, path]))
            .encode()
        ),
    )

    inventory = pypi_inventory.build_inventory(report())

    assert {pin["project"] for pin in inventory["pins"]} == {
        "dinkster",
        "dinkster-api",
        "dinkster-engine",
        "dinkster-supervisor",
        "dinkster-inference",
        "dinkster-token-verifier",
        "dinkster-frontend",
    }
    assert all(inventory["coverage"].values())


def test_build_inventory_refuses_unresolved_retention_input() -> None:
    incomplete = report()
    incomplete["unresolvedConsumerPins"] = [{"consumer": "missing"}]
    with pytest.raises(ValueError, match="incomplete"):
        pypi_inventory.build_inventory(incomplete)


def test_build_inventory_refuses_missing_retained_publication_class(
    monkeypatch,
) -> None:
    incomplete = report()
    incomplete["keep"] = ["engine/current", "base/linux/current"]
    monkeypatch.setattr(
        pypi_inventory,
        "release_manifest",
        lambda entry: (b"{}", {"format": "dinkster.engine/1"}),
    )
    monkeypatch.setattr(
        pypi_inventory, "release_asset", lambda *args: b'{"format":"dinkster.base/1"}'
    )
    monkeypatch.setattr(pypi_inventory, "github_file", lambda *args: b"[project]\n")
    with pytest.raises(ValueError, match="Required retained publication class"):
        pypi_inventory.build_inventory(incomplete)
