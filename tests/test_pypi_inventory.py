from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pypi_inventory
from bundles import manifest_id


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


@pytest.mark.parametrize("changed_identity", [None, "engine", "base", "tools"])
def test_build_inventory_collects_retained_and_current_pins(
    monkeypatch, changed_identity
) -> None:
    engine = {
        "format": "dinkster.engine/1",
        "workspaceWheels": ["dinkster-1.0.0.dev1-py3-none-any.whl"],
        "frontend": {"version": "0.8.0.dev3"},
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
    base = json.loads(
        (Path(__file__).parent / "fixtures/base/linux-cpu.json").read_bytes()
    )
    base["wheels"] = [{"name": "dinkster-api", "version": "1.0.0.dev5"}]
    base["id"] = manifest_id(base)
    assets = {
        (pypi_inventory.ENGINE_REPOSITORY, 1, "release-manifest.json"): engine,
        ("Kosinkadink/Dinkster-Environments", 2, "manifest.json"): base,
        (pypi_inventory.ENGINE_REPOSITORY, 3, "tools-manifest.json"): {
            "format": "dinkster.tools/1",
            "version": "0.4.0.dev2",
        },
    }
    files = {
        (pypi_inventory.ENGINE_REPOSITORY, "scripts/release_sources.json"): {
            "version": "1.0.0.dev6"
        },
        (pypi_inventory.DESKTOP_REPOSITORY, "scripts/frontend-release.json"): {
            "version": "0.9.0.dev4"
        },
        (pypi_inventory.ENGINE_REPOSITORY, "uv.lock"): {
            "text": '[[package]]\nname = "dinkster-supervisor"\n'
            'version = "1.0.0.dev7"\n'
            '[[package]]\nname = "dinkster-token-verifier"\nversion = "0.5.0.dev8"\n'
        },
        ("Kosinkadink/dinkster-nodes", "uv.lock"): {"text": "package = []\n"},
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

    retained = report()
    for entry in retained["publications"]:
        repository = (
            "Kosinkadink/Dinkster-Environments"
            if entry["scope"] == "base"
            else pypi_inventory.ENGINE_REPOSITORY
        )
        name = {
            "engine": "release-manifest.json",
            "base": "manifest.json",
            "tools": "tools-manifest.json",
        }[entry["scope"]]
        value = assets[repository, entry["releaseId"], name]
        entry["identity"] = (
            value["id"]
            if entry["scope"] == "base"
            else "sha256:" + hashlib.sha256(json.dumps(value).encode()).hexdigest()
        )
        if entry["scope"] == changed_identity:
            entry["identity"] = "sha256:" + "0" * 64
    if changed_identity:
        with pytest.raises(ValueError, match="Invalid .* manifest"):
            pypi_inventory.build_inventory(retained)
        return
    inventory = pypi_inventory.build_inventory(retained)

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
    assert {(pin["project"], pin["version"]) for pin in inventory["pins"]} >= {
        ("dinkster-frontend", "0.8.0.dev3"),
        ("dinkster-frontend", "0.9.0.dev4"),
        ("dinkster-frontend", "1.0.0.dev6"),
        ("dinkster-supervisor", "0.4.0.dev2"),
        ("dinkster-token-verifier", "0.5.0.dev8"),
    }


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
