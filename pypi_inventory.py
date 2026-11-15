"""Build the complete protected-wheel inventory for PyPI cleanup reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name, parse_wheel_filename

from pypi_cleanup import PROJECTS, REQUIRED_SOURCE_KINDS

ENGINE_REPOSITORY = "Kosinkadink/Dinkster"
DESKTOP_REPOSITORY = "Kosinkadink/Dinkster-Desktop"


def source(identifier: str, kind: str, location: str, data: bytes) -> dict[str, str]:
    return {
        "id": identifier,
        "kind": kind,
        "location": location,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def github_asset(repository: str, asset_id: int) -> bytes:
    return subprocess.check_output(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/octet-stream",
            f"repos/{repository}/releases/assets/{asset_id}",
        ],
        timeout=60,
    )


def github_file(repository: str, path: str) -> bytes:
    return subprocess.check_output(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github.raw+json",
            f"repos/{repository}/contents/{path}?ref=main",
        ],
        timeout=60,
    )


def release_asset(repository: str, release_id: int, name: str) -> bytes:
    release = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repository}/releases/{release_id}"], timeout=60
        )
    )
    matches = [asset for asset in release["assets"] if asset["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous release asset: {repository} {name}")
    asset = matches[0]
    data = github_asset(repository, asset["id"])
    if asset.get("digest") != "sha256:" + hashlib.sha256(data).hexdigest():
        raise ValueError(f"Release asset digest mismatch: {repository} {name}")
    return data


def add_wheel(
    pins: list[dict[str, str]], source_id: str, name: str, version: str
) -> None:
    project = canonicalize_name(name)
    if project in PROJECTS:
        pins.append({"project": project, "version": version, "source": source_id})


def add_manifest_wheels(
    pins: list[dict[str, str]], source_id: str, manifest: dict[str, Any]
) -> None:
    for filename in manifest.get("workspaceWheels", []):
        name, version, _, _ = parse_wheel_filename(filename)
        add_wheel(pins, source_id, str(name), str(version))
    for dependency in manifest.get("dependencies", {}).values():
        for wheel in dependency.get("wheels", []):
            add_wheel(pins, source_id, wheel["name"], wheel["version"])
    for wheel in manifest.get("wheels", []):
        add_wheel(pins, source_id, wheel["name"], wheel["version"])
    if "frontend" in manifest:
        add_wheel(pins, source_id, "dinkster-frontend", manifest["frontend"]["version"])


def release_manifest(entry: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    data = release_asset(ENGINE_REPOSITORY, entry["releaseId"], "release-manifest.json")
    value = json.loads(data)
    if (
        value.get("format") != "dinkster.engine/1"
        or entry["identity"] != "sha256:" + hashlib.sha256(data).hexdigest()
    ):
        raise ValueError(f"Invalid engine release manifest: {entry['tag']}")
    return data, value


def build_inventory(report: dict[str, Any]) -> dict[str, Any]:
    """Extract protected Dinkster package versions from retained inputs."""
    if report["unresolvedConsumerPins"] or report["unclassifiedDirectories"]:
        raise ValueError(
            "Retention inventory is incomplete; refusing PyPI cleanup inventory"
        )
    entries = {entry["path"]: entry for entry in report["publications"]}
    kept = [entries[path] for path in report["keep"]]
    if not {"engine", "base", "tools"} <= {entry["scope"] for entry in kept}:
        raise ValueError("Required retained publication class is unavailable")
    sources: list[dict[str, str]] = []
    coverage: dict[str, list[str]] = {kind: [] for kind in REQUIRED_SOURCE_KINDS}
    pins: list[dict[str, str]] = []

    for entry in kept:
        if entry["scope"] != "engine":
            continue
        data, manifest = release_manifest(entry)
        identifier = "engine:" + entry["tag"]
        sources.append(
            source(identifier, "retained-engine-manifest", entry["tag"], data)
        )
        coverage["retained-engine-manifest"].append(identifier)
        add_manifest_wheels(pins, identifier, manifest)
    for entry in kept:
        if entry["scope"] == "base":
            identifier = "base:" + entry["tag"]
            data = release_asset(
                "Kosinkadink/Dinkster-Environments", entry["releaseId"], "manifest.json"
            )
            manifest = json.loads(data)
            from bundles import IDENTITY_FIELDS, canonical, manifest_id

            identity = (
                manifest_id(manifest)
                if "archive" in manifest
                else "sha256:"
                + hashlib.sha256(
                    canonical({field: manifest[field] for field in IDENTITY_FIELDS})
                ).hexdigest()
            )
            if (
                manifest.get("format") != "dinkster.base/1"
                or manifest["id"] != identity
                or entry["identity"] != identity
            ):
                raise ValueError(f"Invalid retained base manifest: {entry['tag']}")
            sources.append(
                source(identifier, "retained-base-manifest", entry["tag"], data)
            )
            coverage["retained-base-manifest"].append(identifier)
            add_manifest_wheels(pins, identifier, manifest)
        elif entry["scope"] == "tools":
            identifier = "tools:" + entry["tag"]
            data = release_asset(
                ENGINE_REPOSITORY, entry["releaseId"], "tools-manifest.json"
            )
            manifest = json.loads(data)
            if (
                manifest.get("format") != "dinkster.tools/1"
                or entry["identity"] != "sha256:" + hashlib.sha256(data).hexdigest()
            ):
                raise ValueError(f"Invalid retained tools manifest: {entry['tag']}")
            sources.append(
                source(identifier, "retained-tools-manifest", entry["tag"], data)
            )
            coverage["retained-tools-manifest"].append(identifier)
            add_wheel(pins, identifier, "dinkster-supervisor", manifest["version"])

    for repository, path in (
        (ENGINE_REPOSITORY, "scripts/release_sources.json"),
        (DESKTOP_REPOSITORY, "scripts/frontend-release.json"),
    ):
        data = github_file(repository, path)
        identifier = "frontend:" + repository.rsplit("/", 1)[1]
        sources.append(
            source(
                identifier, "retained-frontend-manifest", repository + "/" + path, data
            )
        )
        coverage["retained-frontend-manifest"].append(identifier)
        add_wheel(pins, identifier, "dinkster-frontend", json.loads(data)["version"])

    for repository, path in (
        (ENGINE_REPOSITORY, "uv.lock"),
        ("Kosinkadink/dinkster-nodes", "uv.lock"),
    ):
        data = github_file(repository, path)
        identifier = "lock:" + repository.rsplit("/", 1)[1]
        sources.append(
            source(identifier, "dependency-lock", repository + "/" + path, data)
        )
        coverage["dependency-lock"].append(identifier)
        document = tomllib.loads(data.decode())
        for package in document["package"]:
            add_wheel(pins, identifier, package["name"], package["version"])
    desktop = github_file(
        DESKTOP_REPOSITORY, "packages/desktop/src/backend-release.json"
    )
    identifier = "consumer:desktop-backend"
    sources.append(
        source(
            identifier,
            "current-consumer-pin",
            "Dinkster-Desktop/backend-release.json",
            desktop,
        )
    )
    coverage["current-consumer-pin"].append(identifier)
    add_wheel(pins, identifier, "dinkster", json.loads(desktop)["version"])
    if any(not coverage[kind] for kind in coverage):
        raise ValueError("Required retained publication class is unavailable")
    return {
        "format": "dinkster.pypi-cleanup-inventory/1",
        "sources": sources,
        "coverage": coverage,
        "pins": pins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = build_inventory(
        json.loads(args.retention_report.read_text(encoding="utf-8"))
    )
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
