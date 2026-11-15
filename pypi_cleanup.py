"""List manually removable PyPI development wheels from a complete pin snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from packaging.version import Version

PROJECTS = (
    "dinkster",
    "dinkster-api",
    "dinkster-engine",
    "dinkster-supervisor",
    "dinkster-inference",
    "dinkster-token-verifier",
    "dinkster-frontend",
)
REQUIRED_SOURCE_KINDS = (
    "retained-engine-manifest",
    "retained-base-manifest",
    "retained-tools-manifest",
    "retained-frontend-manifest",
    "dependency-lock",
    "current-consumer-pin",
)


def read_inventory(path: Path) -> dict[str, Any]:
    """Read a complete, content-addressed snapshot of all protected wheel pins."""
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(inventory, dict)
        or inventory.get("format") != "dinkster.pypi-cleanup-inventory/1"
    ):
        raise ValueError("Unsupported PyPI cleanup inventory format")
    sources = inventory.get("sources")
    coverage = inventory.get("coverage")
    pins = inventory.get("pins")
    if (
        not isinstance(sources, list)
        or not isinstance(coverage, dict)
        or not isinstance(pins, list)
    ):
        raise ValueError("PyPI cleanup inventory requires sources, coverage, and pins")
    source_ids: set[str] = set()
    source_by_id: dict[str, dict[str, str]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Invalid protected-pin source")
        identifier = source.get("id")
        kind = source.get("kind")
        location = source.get("location")
        digest = source.get("sha256")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[A-Za-z0-9._:/-]+", identifier)
            or identifier in source_ids
            or kind not in REQUIRED_SOURCE_KINDS
            or not isinstance(location, str)
            or not location
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("Invalid protected-pin source")
        source_ids.add(identifier)
        source_by_id[identifier] = {
            "id": identifier,
            "kind": kind,
            "location": location,
            "sha256": digest,
        }
    for kind in REQUIRED_SOURCE_KINDS:
        identifiers = coverage.get(kind)
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or any(
                not isinstance(identifier, str)
                or identifier not in source_by_id
                or source_by_id[identifier]["kind"] != kind
                for identifier in identifiers
            )
        ):
            raise ValueError(f"Protected-pin coverage is incomplete: {kind}")
    protected: dict[str, list[dict[str, str]]] = {project: [] for project in PROJECTS}
    for pin in pins:
        if not isinstance(pin, dict):
            raise ValueError("Invalid protected wheel pin")
        project = pin.get("project")
        version = pin.get("version")
        source = pin.get("source")
        if (
            project not in protected
            or not isinstance(version, str)
            or source not in source_by_id
        ):
            raise ValueError("Invalid protected wheel pin")
        try:
            Version(version)
        except ValueError as error:
            raise ValueError(f"Invalid protected wheel version: {version}") from error
        protected[project].append({"version": version, "source": source})
    return {"sources": source_by_id, "protected": protected}


def pypi_releases(project: str) -> dict[str, list[dict[str, Any]]]:
    """Read the public PyPI release inventory without using any credentials."""
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{project}/json", timeout=60
    ) as response:
        payload = json.loads(response.read())
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        raise ValueError(f"Invalid PyPI release inventory: {project}")
    return releases


def wheel_versions(releases: dict[str, list[dict[str, Any]]]) -> dict[Version, str]:
    """Keep exact release spellings only where PyPI retains at least one wheel."""
    result: dict[Version, str] = {}
    for spelling, files in releases.items():
        if not isinstance(spelling, str) or not isinstance(files, list):
            raise ValueError("Invalid PyPI release entry")
        try:
            version = Version(spelling)
        except ValueError as error:
            raise ValueError(f"Invalid PyPI version: {spelling}") from error
        if not any(
            isinstance(file, dict) and file.get("packagetype") == "bdist_wheel"
            for file in files
        ):
            continue
        previous = result.get(version)
        if previous is not None and previous != spelling:
            raise ValueError(f"Ambiguous normalized PyPI version: {spelling}")
        result[version] = spelling
    return result


def plan_project(
    project: str,
    releases: dict[str, list[dict[str, Any]]],
    protected: list[dict[str, str]],
) -> dict[str, Any]:
    """Select development wheels older than the previous stable boundary."""
    versions = wheel_versions(releases)
    stable = sorted(version for version in versions if not version.is_prerelease)
    protected_versions: dict[Version, list[str]] = {}
    for pin in protected:
        protected_versions.setdefault(Version(pin["version"]), []).append(pin["source"])
    result: dict[str, Any] = {
        "stableVersions": [versions[version] for version in stable],
        "protectedVersions": [
            {
                "version": pin["version"],
                "source": pin["source"],
            }
            for pin in protected
        ],
        "candidates": [],
    }
    if len(stable) < 2:
        result["reason"] = "Fewer than two stable wheel releases exist."
        return result
    previous_stable = stable[-2]
    result["currentStable"] = versions[stable[-1]]
    result["previousStable"] = versions[previous_stable]
    result["candidates"] = [
        {
            "version": versions[version],
            "protectedBy": protected_versions.get(version, []),
        }
        for version in sorted(versions)
        if version.is_devrelease
        and version < previous_stable
        and version not in protected_versions
    ]
    return result


def plan(
    inventory: dict[str, Any],
    releases_by_project: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Produce the read-only deletion listing and its pin-source evidence."""
    if set(releases_by_project) != set(PROJECTS):
        raise ValueError("PyPI inventory must cover every protected project")
    return {
        "format": "dinkster.pypi-cleanup-report/1",
        "action": "manual-pypi-web-deletion-only",
        "sources": sorted(
            inventory["sources"].values(), key=lambda source: source["id"]
        ),
        "projects": {
            project: plan_project(
                project, releases_by_project[project], inventory["protected"][project]
            )
            for project in PROJECTS
        },
    }


def markdown(report: dict[str, Any]) -> str:
    """Render a durable owner-facing deletion checklist from the JSON report."""
    lines = [
        "# PyPI development wheel cleanup",
        "",
        "This is a read-only listing. Delete only the listed versions through the PyPI",
        "web UI; do not yank them.",
        "",
        "## Candidates",
        "",
    ]
    for project, details in report["projects"].items():
        candidates = details["candidates"]
        if candidates:
            lines.extend([f"### {project}", ""])
            lines.extend(f"- `{candidate['version']}`" for candidate in candidates)
            lines.append("")
        else:
            reason = details.get("reason", "No removable development wheel versions.")
            lines.extend([f"### {project}", "", f"- {reason}", ""])
    lines.extend(["## Evidence", ""])
    for source in report["sources"]:
        lines.append(
            f"- `{source['kind']}` `{source['id']}`: `{source['location']}` "
            f"sha256:{source['sha256']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    inventory = read_inventory(args.inventory)
    report = plan(inventory, {project: pypi_releases(project) for project in PROJECTS})
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
