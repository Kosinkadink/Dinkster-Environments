"""Plan channel-aware release retention from an authenticated bucket inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from botocore.exceptions import ClientError
from packaging.version import Version


def consumer_pins() -> dict:
    """Read current public consumer inputs, never a cached last-good pin set."""
    sources = {}

    def document(repository: str, path: str) -> dict:
        endpoint = f"repos/Kosinkadink/{repository}/contents/{path}?ref=main"
        data = subprocess.check_output(
            ["gh", "api", "-H", "Accept: application/vnd.github.raw+json", endpoint],
            timeout=60,
        )
        sources[endpoint] = hashlib.sha256(data).hexdigest()
        value = json.loads(data)
        if not isinstance(value, dict) or not value:
            raise ValueError(f"Invalid consumer pin source: {endpoint}")
        return value

    desktop = document(
        "Dinkster-Desktop", "packages/desktop/src/environment-tools-release.json"
    )
    registered = document("Dinkster-Environments", "consumer-pins.json")
    if registered["format"] != "dinkster.consumer-pins/1":
        raise ValueError("Unsupported consumer pin format")
    non_release = registered.get("nonReleaseConsumers", {})
    for name, fixture in non_release.items():
        if (
            not re.fullmatch(r"Kosinkadink/[A-Za-z0-9_.-]+", fixture["repository"])
            or not re.fullmatch(r"[0-9a-f]{40}", fixture["commit"])
            or not re.fullmatch(r"[0-9a-f]{64}", fixture["sha256"])
            or fixture["path"] != "scripts/release-packs.lock"
            or fixture["source"] != "registry:https://registry.dinkster.org"
            or fixture["engineReleaseDependency"] is not False
            or fixture["baseReleaseDependency"] is not False
        ):
            raise ValueError(f"Invalid non-release consumer fixture: {name}")
    desktop_tools = "label" in desktop
    if desktop_tools:
        manifest = desktop["manifest"]
        artifacts = desktop["artifacts"]
        if (
            desktop["repository"] != "Kosinkadink/Dinkster"
            or not re.fullmatch(r"[0-9a-f]{40}", desktop["commit"])
            or desktop["releaseTag"] != "tools-" + desktop["label"]
            or not valid_tools_publication_label(desktop["label"])
            or not valid_tools_file(manifest, "tools-manifest.json")
            or not isinstance(artifacts, list)
            or not all(isinstance(row, dict) for row in artifacts)
            or {row["name"] for row in artifacts} != TOOLS_ARCHIVES
            or any(not valid_tools_file(row) for row in artifacts)
        ):
            raise ValueError("Invalid Desktop tools pin")
        pins = [
            {
                "consumer": "Dinkster-Desktop",
                "repository": desktop["repository"],
                "scope": "tools",
                "tag": desktop["releaseTag"],
                "label": desktop["label"],
                "identity": "sha256:" + manifest["sha256"],
            }
        ]
    else:
        pins = []
    for name, pin in ([] if desktop_tools else [("Dinkster-Desktop", desktop)]) + list(
        registered["consumers"].items()
    ):
        if pin["repository"] != "Kosinkadink/Dinkster" or not re.fullmatch(
            r"(?:main-|v)[A-Za-z0-9._+-]+", pin["releaseTag"]
        ):
            raise ValueError(f"Invalid consumer release pin: {name}")
        pins.append(
            {
                "consumer": name,
                "repository": pin["repository"],
                "tag": pin["releaseTag"],
                "status": pin.get("status", "active"),
            }
        )
    if not desktop_tools:
        digest = desktop["manifest"]["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid Desktop manifest identity")
        pins[0]["identity"] = "sha256:" + digest
    bases = document("Dinkster", "scripts/base_bundles.json")
    for variant, identity in bases.items():
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant) or not re.fullmatch(
            r"[0-9a-f]{64}", identity
        ):
            raise ValueError("Invalid consumer base pin")
        pins.append(
            {
                "consumer": "Dinkster",
                "repository": "Kosinkadink/Dinkster-Environments",
                "variant": variant,
                "identity": "sha256:" + identity,
            }
        )
    return {"sources": sources, "pins": pins, "nonReleaseConsumers": non_release}


TOOLS_ARCHIVES = {
    "dinkster-tools-linux-x86_64.zip",
    "dinkster-tools-windows-x86_64.zip",
    "dinkster-tools-macos-arm64.zip",
}


def valid_tools_file(value: object, name: str | None = None) -> bool:
    return (
        isinstance(value, dict)
        and (name is None or value.get("name") == name)
        and type(value.get("size")) is int
        and value["size"] > 0
        and bool(re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")))
    )


def valid_tools_label(label: object) -> bool:
    if not isinstance(label, str):
        return False
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-([A-Za-z0-9._+-]+)", label)
    if match is None:
        return False
    try:
        datetime.strptime(match[1], "%Y-%m-%d")
        return Version(match[2]).local is None
    except ValueError:
        return False


def valid_tools_publication_label(label: object) -> bool:
    return (
        isinstance(label, str)
        and bool(re.fullmatch(r".+-[0-9a-f]{8}", label))
        and valid_tools_label(label.rsplit("-", 1)[0])
    )


def tools_channel_path(reference: object, public_base_url: str) -> str:
    if not isinstance(reference, dict) or set(reference) != {
        "label",
        "manifest",
        "sha256",
        "release",
    }:
        raise ValueError("Invalid channel tools reference")
    label = reference["label"]
    if not valid_tools_publication_label(label) or not re.fullmatch(
        r"[0-9a-f]{64}", reference["sha256"]
    ):
        raise ValueError("Invalid channel tools identity")
    base = urlsplit(public_base_url)
    manifest = urlsplit(reference["manifest"])
    release = urlsplit(reference["release"])
    if (
        base.scheme != "https"
        or base.netloc == ""
        or manifest.scheme != "https"
        or (manifest.scheme, manifest.netloc) != (base.scheme, base.netloc)
        or manifest.path
        != base.path.rstrip("/") + f"/tools/{label}/tools-manifest.json"
        or (release.scheme, release.netloc, release.path.rstrip("/"))
        != (base.scheme, base.netloc, base.path.rstrip("/") + f"/tools/{label}")
        or manifest.query
        or manifest.fragment
        or release.query
        or release.fragment
    ):
        raise ValueError("Channel tools URL is outside the public mirror")
    return f"tools/{label}"


def pinned_publications(
    entries: list[dict], pins: list[dict]
) -> tuple[set[str], list[dict]]:
    """Resolve exact release pins and base content pins to whole directories."""
    keep = set()
    unresolved = []
    for pin in pins:
        matching = [
            entry
            for entry in entries
            if entry["repository"] == pin["repository"]
            and all(
                entry[field] == pin[field]
                for field in ("tag", "identity", "variant", "scope", "label")
                if field in pin
            )
        ]
        if "variant" in pin:
            labelled = [entry for entry in matching if entry["label"] is not None]
            if len(labelled) > 1:
                raise ValueError(f"Ambiguous consumer base publication: {pin}")
            matching = labelled or matching
        if not matching or not any(
            entry["complete"] and entry["releaseId"] is not None for entry in matching
        ):
            unresolved.append(pin)
            continue
        keep.update(entry["path"] for entry in matching)
    return keep, unresolved


def select_publications(
    entries: list[dict],
    channels: list[dict],
    *,
    engine_count: int,
    predecessors: int,
    verified_aliases: frozenset[str] = frozenset(),
    pinned_paths: frozenset[str] = frozenset(),
    old_naming: bool = False,
) -> dict:
    """Keep physical channel paths, stable engines and distinct base predecessors."""
    if type(engine_count) is not int or engine_count < 0:
        raise ValueError("engineMainReleases must be a nonnegative integer")
    if type(predecessors) is not int or predecessors < 0:
        raise ValueError("basePredecessors must be a nonnegative integer")
    by_path = {entry["path"]: entry for entry in entries}
    if len(by_path) != len(entries):
        raise ValueError("Duplicate publication path")
    keep = set(pinned_paths)
    pointed = []
    for channel in channels:
        if channel.get("format") != "dinkster.channel/1":
            raise ValueError("Unsupported channel format")
        for variant, reference in [(None, channel["engine"]), *channel["base"].items()]:
            identity = reference["release"] if variant is None else reference["id"]
            paths = [
                location["path"].rsplit("/", 1)[0]
                for location in reference["locations"]
                if location["type"] == "mirror"
            ]
            if not paths:
                raise ValueError("Channel has no mirror location")
            suffix = "/release-manifest.json" if variant is None else "/manifest.json"
            if any(
                not location["path"].endswith(suffix)
                for location in reference["locations"]
                if location["type"] == "mirror"
            ):
                raise ValueError("Channel target has an unexpected manifest name")
            for path in paths:
                entry = by_path.get(path)
                if (
                    entry is None
                    or entry["identity"] != identity
                    or entry["variant"] != variant
                    or not entry["complete"]
                ):
                    raise ValueError(f"Channel target is missing or incomplete: {path}")
                keep.add(path)
                pointed.append(entry)
    if not channels:
        raise ValueError("No channels found; refusing retention")
    engines = [entry for entry in entries if entry["variant"] is None]
    keep.update(entry["path"] for entry in entries if not entry["complete"])
    keep.update(entry["path"] for entry in engines if entry["tag"].startswith("v"))
    pointed_variants = {entry["variant"] for entry in pointed}
    keep.update(
        entry["path"]
        for entry in entries
        if entry["variant"] is not None and entry["variant"] not in pointed_variants
    )
    protected = set(keep)
    eligible = sorted(
        (
            entry
            for entry in engines
            if entry["tag"].startswith("main-") and entry["complete"]
        ),
        key=lambda entry: (entry["publishedAt"], entry["path"]),
        reverse=True,
    )
    newest_tags = set(
        list(dict.fromkeys(entry["tag"] for entry in eligible))[:engine_count]
    )
    keep.update(
        entry["path"]
        for entry in eligible
        if entry["tag"] in newest_tags and not entry["thirdPartyWheels"]
    )
    base_variants = {}
    for current in pointed:
        variant = current["variant"]
        if variant is None:
            continue
        # Republishing a base does not create a new content generation.
        generations: dict[str, dict] = {}
        first_published: dict[str, int] = {}
        for entry in entries:
            if entry["variant"] != variant or not entry["complete"]:
                continue
            identity = entry["identity"]
            if (
                identity not in first_published
                or entry["publicationOrder"] < first_published[identity]
            ):
                first_published[identity] = entry["publicationOrder"]
            old = generations.get(entry["identity"])
            if old is None or (bool(entry["label"]), entry["publishedAt"]) > (
                bool(old["label"]),
                old["publishedAt"],
            ):
                generations[entry["identity"]] = entry
        ordered = sorted(
            generations.values(),
            key=lambda entry: (first_published[entry["identity"]], entry["identity"]),
            reverse=True,
        )
        index = next(
            i
            for i, entry in enumerate(ordered)
            if entry["identity"] == current["identity"]
        )
        keep.update(
            entry["path"] for entry in ordered[index + 1 : index + 1 + predecessors]
        )
        details = base_variants.setdefault(
            variant,
            {
                "targets": [],
                "previous": [],
                "pending": [],
                "aliases": [],
            },
        )
        details["targets"].append(current["path"])
        details["previous"].extend(
            entry["path"] for entry in ordered[index + 1 : index + 1 + predecessors]
        )
        details["pending"].extend(entry["path"] for entry in ordered[:index])
        details["aliases"].extend(
            {
                "identity": identity,
                "paths": sorted(
                    entry["path"]
                    for entry in entries
                    if entry["variant"] == variant and entry["identity"] == identity
                ),
                "publicationOrder": first_published[identity],
            }
            for identity in generations
            if sum(
                entry["variant"] == variant and entry["identity"] == identity
                for entry in entries
            )
            > 1
        )
        # A channel still using the hash alias prevents its removal.
        keep.add(generations[current["identity"]]["path"])
        # A newly published generation can precede the channel update.
        keep.update(entry["path"] for entry in ordered[:index])
    for entry in entries:
        if entry["path"].rsplit("/", 1)[-1] != entry["identity"][7:]:
            continue
        aliases = [
            alias
            for alias in entries
            if alias["variant"] == entry["variant"]
            and alias["identity"] == entry["identity"]
            and alias["path"] != entry["path"]
        ]
        if aliases and not any(
            alias["complete"] and alias["path"] in verified_aliases for alias in aliases
        ):
            keep.add(entry["path"])
            protected.add(entry["path"])
    if old_naming:
        keep.difference_update(
            entry["path"]
            for entry in entries
            if entry["complete"]
            and entry["path"] not in protected
            and not entry["tag"].startswith("v")
            and (
                re.fullmatch(r"[0-9a-f]{64}", entry["path"].rsplit("/", 1)[-1])
                or re.fullmatch(
                    r"engine/[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]+\.[0-9]+\.[0-9]+"
                    r"\.dev[0-9]+\+g[0-9a-f]{40}-[0-9a-f]{8}",
                    entry["path"],
                )
            )
        )
    retained_tags = {entry["tag"] for entry in entries if entry["path"] in keep}
    shared_tags = {
        tag: sorted(entry["path"] for entry in entries if entry["tag"] == tag)
        for tag in sorted(retained_tags)
        if sum(entry["tag"] == tag for entry in entries) > 1
    }
    # One GitHub tag cannot be removed for one directory and kept for another.
    keep.update(path for paths in shared_tags.values() for path in paths)
    for details in base_variants.values():
        for field in ("targets", "previous", "pending"):
            details[field] = sorted(set(details[field]))
        aliases_by_id = {alias["identity"]: alias for alias in details["aliases"]}
        details["aliases"] = [
            aliases_by_id[identity] for identity in sorted(aliases_by_id)
        ]
        for alias in details["aliases"]:
            alias["retainedPaths"] = [path for path in alias["paths"] if path in keep]
            alias["deletablePaths"] = [
                path for path in alias["paths"] if path not in keep
            ]
    return {
        "keep": sorted(keep),
        "baseVariants": base_variants,
        "sharedTags": shared_tags,
    }


def github_releases(repository: str) -> list[dict]:
    pages = json.loads(
        subprocess.check_output(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repository}/releases?per_page=100",
            ],
            text=True,
        )
    )
    return [release for page in pages for release in page]


def inventory(client, bucket: str) -> tuple[dict[str, dict], dict[str, bytes]]:
    objects = {
        row["Key"]: row
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket)
        for row in page.get("Contents", [])
    }
    metadata = {
        key: client.get_object(Bucket=bucket, Key=key)["Body"].read()
        for key in objects
        if (key.startswith("channels/") and key.endswith(".json"))
        or (key.startswith("base/") and key.endswith("/index.json"))
        or (key.startswith("engine/") and key.endswith("/release-manifest.json"))
        or (key.startswith("tools/") and key.endswith("/tools-manifest.json"))
        or (key.startswith("tools/") and key.endswith("/SHA256SUMS"))
        or key == "tools/index.json"
        or (
            key.startswith("base/") and key.endswith(("/manifest.json", "/extras.json"))
        )
    }
    return objects, metadata


def publications(
    objects: dict[str, dict], metadata: dict[str, bytes], releases: dict
) -> list[dict]:
    from bundles import IDENTITY_FIELDS, base_release_name, canonical, manifest_id

    result = []
    base_contents: dict[tuple[str, str], bytes] = {}
    for key, data in metadata.items():
        if key.startswith(("channels/", "tools/")) or key.endswith(
            ("/extras.json", "/index.json")
        ):
            continue
        manifest = json.loads(data)
        prefix = key.rsplit("/", 1)[0]
        parts = prefix.split("/")
        variant = parts[1] if parts[0] == "base" else None
        if len(parts) != (3 if variant else 2):
            raise ValueError(f"Unexpected publication path: {key}")
        label = manifest.get("label")
        publication_order = None
        if variant is None:
            identity = "sha256:" + hashlib.sha256(data).hexdigest()
            if manifest["repository"] != "Kosinkadink/Dinkster":
                raise ValueError(f"Unexpected engine repository: {key}")
            stable = not Version(manifest["version"]).is_prerelease
            tag = manifest.get("tag") or (
                "v" + manifest["version"]
                if stable
                else "main-" + (parts[-1] if label else manifest["commit"])
            )
            expected = f"{label}-{identity[7:15]}" if label else identity[7:]
            if parts[-1] != expected or not (
                tag == "v" + manifest["version"]
                or tag == "main-" + (parts[-1] if label else manifest["commit"])
            ):
                raise ValueError(f"Engine name differs from identity: {key}")
            workspace = set(manifest["workspaceWheels"])
            third_party = any(
                row["name"].endswith(".whl") and row["name"] not in workspace
                for row in manifest["artifacts"]
            )
            payloads = [(row["name"], row["size"]) for row in manifest["artifacts"]]
            payloads.append(("SHA256SUMS", None))
            repository = "Kosinkadink/Dinkster"
        else:
            index_key = f"base/{variant}/index.json"
            if index_key not in metadata:
                raise ValueError(f"Missing publication order: {index_key}")
            index = json.loads(metadata[index_key])
            paths = [row["path"] for row in index]
            if len(set(paths)) != len(paths) or prefix not in paths:
                raise ValueError(f"Missing or ambiguous publication order: {prefix}")
            publication_order = len(paths) - paths.index(prefix)
            identity = (
                manifest_id(manifest)
                if "archive" in manifest
                else (
                    "sha256:"
                    + hashlib.sha256(
                        canonical({field: manifest[field] for field in IDENTITY_FIELDS})
                    ).hexdigest()
                )
            )
            if manifest["id"] != identity or manifest["variant"] != variant:
                raise ValueError(f"Base identity mismatch: {key}")
            if parts[-1] not in (identity[7:], label):
                raise ValueError(f"Base path differs from identity: {key}")
            tag = f"base-{variant}-{parts[-1]}"
            if base_release_name(tag, variant) != parts[-1]:
                raise ValueError(f"Invalid base publication name: {key}")
            archive = (
                manifest["archive"]["parts"]
                if "archive" in manifest
                else manifest["parts"]
            )
            payloads = [(part["filename"], part["size"]) for part in archive]
            payloads.append(("extras.json", None))
            extras_bytes = metadata.get(prefix + "/extras.json")
            if extras_bytes is not None:
                extras = json.loads(extras_bytes)
                if (
                    extras["format"] != "dinkster.extras/1"
                    or extras["base"] != identity
                    or extras["variant"] != variant
                ):
                    raise ValueError(f"Base extras identity mismatch: {key}")
                content = canonical({"parts": archive, "extras": extras})
                previous = base_contents.setdefault((variant, identity), content)
                if previous != content:
                    raise ValueError(f"Base aliases have different payloads: {key}")
            third_party = False
            repository = "Kosinkadink/Dinkster-Environments"
        if any(Path(name).name != name or name in (".", "..") for name, _ in payloads):
            raise ValueError(f"Unsafe payload name: {key}")
        matching = [
            release for release in releases[repository] if release["tag_name"] == tag
        ]
        if len(matching) > 1:
            raise ValueError(f"Ambiguous GitHub release: {tag}")
        release = matching[0] if matching else None
        complete = all(
            (prefix + "/" + name) in objects
            and (size is None or objects[prefix + "/" + name]["Size"] == size)
            for name, size in payloads
        )
        if release and (
            release["draft"]
            or (
                variant is None
                and tag.startswith("main-")
                and not release["prerelease"]
            )
        ):
            complete = False
        result.append(
            {
                "path": prefix,
                "identity": identity,
                "variant": variant,
                "label": label,
                "tag": tag,
                "version": manifest.get("version"),
                "repository": repository,
                "releaseId": release["id"] if release else None,
                "publishedAt": (release or {}).get("published_at")
                or manifest.get("builtAt")
                or objects[key]["LastModified"].isoformat(),
                "publicationOrder": publication_order,
                "commit": manifest.get("commit") if variant is None else None,
                "thirdPartyWheels": third_party,
                "complete": complete,
                "scope": "base" if variant is not None else "engine",
                "size": sum(
                    row["Size"]
                    for name, row in objects.items()
                    if name.startswith(prefix + "/")
                ),
            }
        )
    return result


def tools_publications(
    objects: dict[str, dict],
    metadata: dict[str, bytes],
    releases: list[dict],
    pending_paths: frozenset[str] = frozenset(),
) -> list[dict]:
    """Read independent tools publications and bind every directory to its manifest."""
    index = json.loads(metadata.get("tools/index.json", b"[]"))
    if not isinstance(index, list):
        raise ValueError("Invalid tools index")
    rows = {row.get("path"): row for row in index if isinstance(row, dict)}
    if len(rows) != len(index):
        raise ValueError("Invalid or duplicate tools index path")
    result = []
    for key, data in metadata.items():
        if not key.startswith("tools/") or not key.endswith("/tools-manifest.json"):
            continue
        prefix = key.rsplit("/", 1)[0]
        parts = prefix.split("/")
        if len(parts) != 2:
            raise ValueError(f"Unexpected tools publication path: {key}")
        if prefix in pending_paths:
            continue
        manifest = json.loads(data)
        identity = hashlib.sha256(data).hexdigest()
        label = manifest.get("label")
        artifacts = manifest.get("artifacts")
        if (
            manifest.get("format") != "dinkster.tools/1"
            or manifest.get("repository") != "Kosinkadink/Dinkster"
            or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("commit", ""))
            or not valid_tools_label(label)
            or parts[1] != f"{label}-{identity[:8]}"
            or not isinstance(manifest.get("builtAt"), str)
            or not isinstance(artifacts, list)
            or len(artifacts) != len(TOOLS_ARCHIVES)
            or not all(isinstance(row, dict) for row in artifacts)
            or {row["name"] for row in artifacts} != TOOLS_ARCHIVES
            or any(not valid_tools_file(row) for row in artifacts)
        ):
            raise ValueError(f"Invalid tools manifest: {key}")
        checksums = {
            name: digest
            for digest, name in re.findall(
                r"^([0-9a-f]{64})  ([^\n/]+)$",
                metadata.get(prefix + "/SHA256SUMS", b"").decode("ascii"),
                re.MULTILINE,
            )
        }
        if checksums != {
            "tools-manifest.json": identity,
            **{row["name"]: row["sha256"] for row in artifacts},
        }:
            raise ValueError(f"Tools checksums differ from manifest: {key}")
        try:
            built_at = datetime.fromisoformat(
                manifest["builtAt"].replace("Z", "+00:00")
            )
            offset = built_at.utcoffset()
            if (
                offset is None
                or offset.total_seconds() != 0
                or label
                != (
                    f"{built_at.date().isoformat()}-{Version(manifest['version']).public}"
                )
            ):
                raise ValueError
        except ValueError as error:
            raise ValueError(f"Invalid tools build time: {key}") from error
        expected = {
            "label": parts[1],
            "sha256": identity,
            "commit": manifest["commit"],
            "version": manifest["version"],
            "builtAt": manifest["builtAt"],
            "size": sum(
                row["Size"]
                for name, row in objects.items()
                if name.startswith(prefix + "/")
            ),
            "path": prefix,
        }
        if rows.get(prefix) != expected:
            raise ValueError(f"Tools index differs from manifest: {key}")
        tag = "tools-" + parts[1]
        matching = [release for release in releases if release["tag_name"] == tag]
        if len(matching) > 1:
            raise ValueError(f"Ambiguous GitHub release: {tag}")
        release = matching[0] if matching else None
        payloads = [("tools-manifest.json", None), ("SHA256SUMS", None)] + [
            (row["name"], row["size"]) for row in artifacts
        ]
        complete = (
            all(
                prefix + "/" + name in objects
                and (size is None or objects[prefix + "/" + name]["Size"] == size)
                for name, size in payloads
            )
            and bool(release)
            and not release["draft"]
            and release["prerelease"]
        )
        result.append(
            {
                "path": prefix,
                "identity": "sha256:" + identity,
                "variant": None,
                "label": parts[1],
                "tag": tag,
                "version": manifest["version"],
                "repository": "Kosinkadink/Dinkster",
                "releaseId": release["id"] if release else None,
                "publishedAt": (release or {}).get("published_at")
                or manifest["builtAt"],
                "publicationOrder": None,
                "commit": manifest["commit"],
                "thirdPartyWheels": False,
                "complete": complete,
                "size": sum(
                    row["Size"]
                    for name, row in objects.items()
                    if name.startswith(prefix + "/")
                ),
                "scope": "tools",
            }
        )
    if set(rows) - pending_paths != {entry["path"] for entry in result}:
        raise ValueError("Tools index names a missing publication")
    return result


def select_tools_publications(
    entries: list[dict], channels: list[dict], pinned_paths: frozenset[str], count: int
) -> dict:
    if type(count) is not int or count < 0:
        raise ValueError("toolsReleases must be a nonnegative integer")
    by_path = {entry["path"]: entry for entry in entries}
    keep = set(pinned_paths)
    pointed = []
    public_base_url = os.environ.get("R2_PUBLIC_BASE_URL")
    if not public_base_url and (
        entries or any("tools" in channel for channel in channels)
    ):
        raise ValueError("Tools retention requires R2_PUBLIC_BASE_URL")
    for channel in channels:
        if "tools" not in channel:
            continue
        if public_base_url is None:
            raise ValueError("Tools retention requires R2_PUBLIC_BASE_URL")
        path = tools_channel_path(channel["tools"], public_base_url)
        entry = by_path.get(path)
        if (
            entry is None
            or entry["identity"] != "sha256:" + channel["tools"]["sha256"]
            or entry["label"] != channel["tools"]["label"]
            or not entry["complete"]
        ):
            raise ValueError(f"Channel tools target is missing or incomplete: {path}")
        keep.add(path)
        pointed.append(path)
    keep.update(entry["path"] for entry in entries if not entry["complete"])
    newest = sorted(
        (entry for entry in entries if entry["complete"]),
        key=lambda entry: (entry["publishedAt"], entry["path"]),
        reverse=True,
    )[:count]
    keep.update(entry["path"] for entry in newest)
    return {"keep": sorted(keep), "targets": sorted(pointed)}


def pending_journal(client, bucket: str, scope: str) -> dict | None:
    """Bind recovery records to their owning scope before classifying publications."""
    from bundles import base_release_name

    journal = read_optional(client, bucket, f"retention/{scope}-pending.json")
    if journal is None:
        return None
    if not isinstance(journal, dict) or not isinstance(journal.get("delete"), list):
        raise ValueError(f"Invalid {scope} pending journal")
    paths = set()
    for entry in journal["delete"]:
        try:
            path, identity = entry["path"], entry["identity"]
            parts = path.split("/")
            variant = entry["variant"]
            if (
                entry.get("scope", scope) != scope
                or parts[0] != scope
                or len(parts) != (3 if scope == "base" else 2)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity)
                or path in paths
                or entry["repository"]
                != "Kosinkadink/"
                + ("Dinkster-Environments" if scope == "base" else "Dinkster")
            ):
                raise ValueError
            if scope == "base":
                if (
                    not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant)
                    or parts[1] != variant
                    or parts[2] not in (identity[7:], entry["label"])
                    or base_release_name(entry["tag"], variant) != parts[2]
                ):
                    raise ValueError
            else:
                if variant is not None or not re.fullmatch(
                    r"[0-9a-f]{40}", entry["commit"]
                ):
                    raise ValueError
                if scope == "tools":
                    if (
                        not valid_tools_publication_label(parts[1])
                        or parts[1] != entry["label"]
                        or not parts[1].endswith("-" + identity[7:15])
                        or entry["tag"] != "tools-" + parts[1]
                    ):
                        raise ValueError
                elif parts[1] != (
                    f"{entry['label']}-{identity[7:15]}"
                    if entry["label"]
                    else identity[7:]
                ) or entry["tag"] != "main-" + (
                    parts[1] if entry["label"] else entry["commit"]
                ):
                    raise ValueError
            objects = entry["objects"]
            if not isinstance(objects, list) or (
                not objects
                and not (scope == "base" and entry.get("releaseOnly") is True)
            ):
                raise ValueError
            keys = set()
            for row in objects:
                key = row["Key"]
                if (
                    not key.startswith(path + "/")
                    or len(key.split("/")) != len(parts) + 1
                    or key.rsplit("/", 1)[1] in ("", ".", "..")
                    or key in keys
                    or type(row["Size"]) is not int
                    or row["Size"] < 0
                    or not isinstance(row["ETag"], str)
                    or not row["ETag"]
                ):
                    raise ValueError
                keys.add(key)
            paths.add(path)
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ValueError(f"Invalid {scope} pending journal entry") from error
    return journal


def plan(client, bucket: str, policy: dict, *, old_naming: bool = False) -> dict:
    consumers = consumer_pins()
    objects, metadata = inventory(client, bucket)
    releases = {
        repo: github_releases(repo)
        for repo in ("Kosinkadink/Dinkster", "Kosinkadink/Dinkster-Environments")
    }
    entries = publications(objects, metadata, releases)
    journals = {
        scope: pending_journal(client, bucket, scope)
        for scope in ("engine", "base", "tools")
    }
    tools_journal = journals["tools"]
    entries.extend(
        tools_publications(
            objects,
            metadata,
            releases["Kosinkadink/Dinkster"],
            frozenset(entry["path"] for entry in tools_journal["delete"])
            if tools_journal
            else frozenset(),
        )
    )
    for scope in ("engine", "base", "tools"):
        journal = journals[scope]
        if journal is None:
            continue
        for pending in journal["delete"]:
            path = pending["path"]
            expected = {row["Key"]: row for row in pending["objects"]}
            remaining = {
                key: row for key, row in objects.items() if key.startswith(path + "/")
            }
            if any(
                key not in expected
                or row["ETag"] != expected[key]["ETag"]
                or row["Size"] != expected[key]["Size"]
                for key, row in remaining.items()
            ):
                raise ValueError(f"Pending directory changed since planning: {path}")
            # A partial delete may have removed its manifest or an archive part.
            # Reclassify the original publication against the current channels.
            entries = [entry for entry in entries if entry["path"] != path]
            if remaining:
                restored = {
                    **pending,
                    "scope": scope,
                    "size": sum(row["Size"] for row in remaining.values()),
                }
                if pending["variant"] is not None:
                    index = json.loads(
                        metadata[f"base/{pending['variant']}/index.json"]
                    )
                    paths = [row["path"] for row in index]
                    if path not in paths:
                        raise ValueError(
                            f"Pending base absent from publication order: {path}"
                        )
                    restored["publicationOrder"] = len(paths) - paths.index(path)
                entries.append(restored)
    base_order = read_optional(client, bucket, "retention/base-publication-order.json")
    if base_order is None:
        base_order = {}
    for variant in sorted({entry["variant"] for entry in entries if entry["variant"]}):
        order = base_order.setdefault(variant, [])
        if len(order) != len(set(order)):
            raise ValueError(f"Duplicate base identity in publication order: {variant}")
        # Observe the oldest copy of each identity first. Persist this order
        # before removing aliases, whose new builtAt otherwise reorders indexes.
        variant_entries = sorted(
            (entry for entry in entries if entry["variant"] == variant),
            key=lambda entry: entry["publicationOrder"],
        )
        for entry in variant_entries:
            if entry["identity"] not in order:
                order.append(entry["identity"])
        for entry in variant_entries:
            entry["publicationOrder"] = order.index(entry["identity"]) + 1
    channels = [
        json.loads(data)
        for key, data in metadata.items()
        if key.startswith("channels/")
    ]
    pinned, unresolved = pinned_publications(entries, consumers["pins"])
    selection = select_publications(
        [entry for entry in entries if entry["scope"] != "tools"],
        channels,
        engine_count=policy["engineMainReleases"],
        predecessors=policy["basePredecessors"],
        verified_aliases=frozenset(policy["verifiedAliasPaths"]),
        pinned_paths=frozenset(entry["path"] for entry in entries)
        if unresolved
        else frozenset(pinned),
        old_naming=old_naming,
    )
    tools = [entry for entry in entries if entry["scope"] == "tools"]
    tools_selection = select_tools_publications(
        tools,
        channels,
        frozenset(entry["path"] for entry in tools)
        if unresolved
        else frozenset(pinned),
        policy.get("toolsReleases", 10),
    )
    keep = set(selection["keep"]) | set(tools_selection["keep"])
    release_only = []
    if old_naming and not unresolved:
        from bundles import IDENTITY_FIELDS, canonical, manifest_id

        repository = "Kosinkadink/Dinkster-Environments"
        for release in releases[repository]:
            match = re.fullmatch(
                r"base-([a-z0-9]+(?:-[a-z0-9]+)*)-([0-9a-f]{64})",
                release["tag_name"],
            )
            if not match or release["draft"]:
                continue
            variant, digest = match.groups()
            path = f"base/{variant}/{digest}"
            if any(entry["path"] == path for entry in entries):
                continue
            if any(key.startswith(path + "/") for key in objects):
                raise ValueError(f"Unclassified old-name directory: {path}")
            identity = "sha256:" + digest
            if any(
                pin["repository"] == repository
                and (
                    pin.get("tag") == release["tag_name"]
                    or (
                        pin.get("variant") == variant
                        and pin.get("identity") == identity
                    )
                )
                for pin in consumers["pins"]
            ):
                continue
            assets = [
                asset for asset in release["assets"] if asset["name"] == "manifest.json"
            ]
            if len(assets) != 1:
                raise ValueError(f"Missing or ambiguous release manifest: {path}")
            manifest = json.loads(
                subprocess.check_output(
                    [
                        "gh",
                        "api",
                        "-H",
                        "Accept: application/octet-stream",
                        f"repos/{repository}/releases/assets/{assets[0]['id']}",
                    ],
                    timeout=60,
                )
            )
            actual = (
                manifest_id(manifest)
                if "archive" in manifest
                else "sha256:"
                + hashlib.sha256(
                    canonical({field: manifest[field] for field in IDENTITY_FIELDS})
                ).hexdigest()
            )
            if (
                manifest["id"] != identity
                or actual != identity
                or manifest["variant"] != variant
            ):
                raise ValueError(f"Release manifest identity mismatch: {path}")
            release_only.append(
                {
                    "path": path,
                    "identity": identity,
                    "variant": variant,
                    "label": digest,
                    "tag": release["tag_name"],
                    "repository": repository,
                    "releaseId": release["id"],
                    "commit": None,
                    "scope": "base",
                    "size": 0,
                    "releaseOnly": True,
                }
            )
    return {
        "mode": "dry-run",
        "oldNaming": old_naming,
        "consumerPins": consumers,
        "unresolvedConsumerPins": unresolved,
        "engineDeletionEnabled": policy.get("engineDeletionEnabled") is True,
        "toolsDeletionEnabled": policy.get("toolsDeletionEnabled") is True,
        "bucketBytes": sum(row["Size"] for row in objects.values()),
        "delete": [entry for entry in entries if entry["path"] not in keep]
        + release_only,
        "keep": sorted(keep),
        "baseVariants": selection["baseVariants"],
        "tools": tools_selection,
        "basePublicationOrder": base_order,
        "sharedTags": selection["sharedTags"],
        "reclaimedBytes": sum(
            entry["size"] for entry in entries if entry["path"] not in keep
        ),
        "channels": {
            key: json.loads(data)
            for key, data in metadata.items()
            if key.startswith("channels/")
        },
        "publications": entries,
        "unclassifiedDirectories": sorted(
            {
                "/".join(key.split("/")[: 3 if key.startswith("base/") else 2])
                for key in objects
                if key.startswith(("engine/", "base/", "tools/"))
                and len(key.split("/")) >= (4 if key.startswith("base/") else 3)
            }
            - {entry["path"] for entry in entries}
        ),
    }


@contextmanager
def publication_lock(client, bucket: str):
    """Serialize both publishers; never steal a lock from a possibly live writer."""
    key = "retention/publication-lock.json"
    owner = str(uuid.uuid4())
    body = json.dumps(
        {
            "owner": owner,
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "run": os.environ.get("GITHUB_RUN_ID"),
        }
    ).encode("ascii")
    # Seven serialized 639-second publishers need over 3.7 hours with 3x margin.
    deadline = time.monotonic() + 4 * 60 * 60
    while True:
        try:
            current = client.get_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise
            condition = {"IfNoneMatch": "*"}
        else:
            document = json.loads(current["Body"].read())
            if document["owner"] is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"Publication lock is held: {document}")
                time.sleep(min(10, remaining))
                continue
            condition = {"IfMatch": current["ETag"]}
        try:
            held = client.put_object(Bucket=bucket, Key=key, Body=body, **condition)
            break
        except ClientError as error:
            if error.response["ResponseMetadata"]["HTTPStatusCode"] != 412:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Could not acquire publication lock") from error
            time.sleep(min(10, remaining))
    try:
        yield
    finally:
        client.put_object(
            Bucket=bucket, Key=key, Body=b'{"owner":null}\n', IfMatch=held["ETag"]
        )


def read_optional(client, bucket: str, key: str) -> dict | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            return None
        raise
    return json.loads(response["Body"].read())


def apply(client, bucket: str, policy: dict, scope: str, report: dict) -> None:
    """Resume a durable directory plan across partial GitHub or bucket failures."""
    if report["unresolvedConsumerPins"]:
        raise ValueError("Unresolved consumer pins; refusing retention")
    if scope == "engine" and policy.get("engineDeletionEnabled") is not True:
        raise ValueError("Engine deletion is disabled")
    if scope == "tools" and policy.get("toolsDeletionEnabled") is not True:
        raise ValueError("Tools deletion is disabled")
    approval_fields = (
        ("toolsDeletionApproval", "toolsDryRunListing")
        if scope == "tools"
        else ("deletionApproval", "dryRunListing")
    )
    approval_issue = "633" if scope == "tools" else "623"
    for field in approval_fields:
        if not re.fullmatch(
            rf"https://github\.com/Kosinkadink/comfy-vibe-station/issues/{approval_issue}#issuecomment-[0-9]+",
            policy.get(field) or "",
        ):
            raise ValueError(f"Deletion requires the recorded {field}")
    repository = "Kosinkadink/" + (
        "Dinkster" if scope in ("engine", "tools") else "Dinkster-Environments"
    )
    if (
        os.environ.get("GITHUB_REPOSITORY") != repository
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
    ):
        raise ValueError(
            "Retention application is restricted to the owning repository main"
        )
    if consumer_pins() != report["consumerPins"]:
        raise ValueError("Consumer pins changed since planning; refusing retention")
    if scope == "base":
        client.put_object(
            Bucket=bucket,
            Key="retention/base-publication-order.json",
            Body=(json.dumps(report["basePublicationOrder"]) + "\n").encode("ascii"),
            ContentType="application/json",
        )
    journal_key = f"retention/{scope}-pending.json"
    journal = pending_journal(client, bucket, scope)
    objects, _ = inventory(client, bucket)
    if journal is None:
        entries = [
            entry
            for entry in report["delete"]
            if entry["repository"] == repository and entry["scope"] == scope
        ]
        for entry in entries:
            entry["objects"] = [
                {"Key": key, "Size": row["Size"], "ETag": row["ETag"]}
                for key, row in objects.items()
                if key.startswith(entry["path"] + "/")
            ]
        journal = {
            "approval": policy[approval_fields[0]],
            "dryRunListing": policy[approval_fields[1]],
            "delete": entries,
        }
        client.put_object(
            Bucket=bucket,
            Key=journal_key,
            Body=(json.dumps(journal) + "\n").encode("ascii"),
        )
    protected = {
        location["path"].rsplit("/", 1)[0]
        for channel in report["channels"].values()
        for reference in [channel["engine"], *channel["base"].values()]
        for location in reference["locations"]
        if location["type"] == "mirror"
    }
    public_base_url = os.environ.get("R2_PUBLIC_BASE_URL")
    for channel in report["channels"].values():
        if "tools" in channel:
            if public_base_url is None:
                raise ValueError("Tools retention requires R2_PUBLIC_BASE_URL")
            protected.add(tools_channel_path(channel["tools"], public_base_url))
    retained_tags = {
        publication["tag"]
        for publication in report["publications"]
        if publication["path"] in report["keep"]
    }
    for entry in journal["delete"]:
        path, tag = entry["path"], entry["tag"]
        if consumer_pins() != report["consumerPins"]:
            raise ValueError("Consumer pins changed since planning; refusing retention")
        if (
            entry["repository"] != repository
            or path in protected
            or path in report["keep"]
            or tag in retained_tags
            or not path.startswith(scope + "/")
            or not re.fullmatch(r"[A-Za-z0-9._+/-]+", path)
            or ".." in path.split("/")
            or not tag.startswith(
                {"engine": "main-", "base": "base-", "tools": "tools-"}[scope]
            )
        ):
            raise ValueError(f"Pending deletion is no longer safe: {path}")
        keys = entry["objects"]
        if entry.get("releaseOnly") and (
            keys
            or any(key.startswith(path + "/") for key in objects)
            or any(
                pin["repository"] == repository
                and (
                    pin.get("tag") == tag
                    or (
                        pin.get("variant") == entry["variant"]
                        and pin.get("identity") == entry["identity"]
                    )
                )
                for pin in report["consumerPins"]["pins"]
            )
        ):
            raise ValueError(
                f"Release-only directory is no longer empty or unpinned: {path}"
            )
        expected = {row["Key"]: row for row in keys}
        if any(not key.startswith(path + "/") for key in expected) or any(
            key not in expected
            or row["ETag"] != expected[key]["ETag"]
            or row["Size"] != expected[key]["Size"]
            for key, row in objects.items()
            if key.startswith(path + "/")
        ):
            raise ValueError(f"Pending directory changed since planning: {path}")
        index_key = (
            "engine/index.json"
            if scope == "engine"
            else "tools/index.json"
            if scope == "tools"
            else "base/" + entry["variant"] + "/index.json"
        )
        index = json.loads(
            client.get_object(Bucket=bucket, Key=index_key)["Body"].read()
        )
        release = next(
            (r for r in github_releases(repository) if r["tag_name"] == tag), None
        )
        if release is not None:
            if (
                release["draft"]
                or release["id"] != entry["releaseId"]
                or (scope in ("engine", "tools") and not release["prerelease"])
            ):
                raise ValueError(f"GitHub release changed since planning: {tag}")
            subprocess.run(
                [
                    "gh",
                    "api",
                    "--method",
                    "DELETE",
                    f"repos/{repository}/releases/{release['id']}",
                ],
                check=True,
            )
            print(f"GitHub release deleted: {repository} {tag}", flush=True)
        else:
            print(f"GitHub release already absent: {repository} {tag}", flush=True)
        # Resolve refs separately: a release's target_commitish may be a branch.
        refs = json.loads(
            subprocess.check_output(
                ["gh", "api", f"repos/{repository}/git/matching-refs/tags/{tag}"],
                text=True,
            )
        )
        exact = next((ref for ref in refs if ref["ref"] == "refs/tags/" + tag), None)
        target = exact["object"] if exact else None
        visited = set()
        while target and target["type"] == "tag":
            if target["sha"] in visited:
                raise ValueError(f"Cyclic annotated tag: {tag}")
            visited.add(target["sha"])
            target = json.loads(
                subprocess.check_output(
                    ["gh", "api", f"repos/{repository}/git/tags/{target['sha']}"],
                    text=True,
                )
            )["object"]
        # Base names encode the verified full content id or manifest label.
        # An engine tag additionally names a specific source commit.
        if (
            target
            and target["type"] == "commit"
            and (scope == "base" or target["sha"] == entry["commit"])
        ):
            subprocess.run(
                [
                    "gh",
                    "api",
                    "--method",
                    "DELETE",
                    f"repos/{repository}/git/refs/tags/{tag}",
                ],
                check=True,
            )
            print(f"GitHub tag deleted: {repository} {tag}", flush=True)
        elif exact:
            print(f"GitHub tag unmatched, retained: {repository} {tag}", flush=True)
        else:
            print(f"GitHub tag already absent: {repository} {tag}", flush=True)
        for start in range(0, len(keys), 1000):
            response = client.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [
                        {"Key": row["Key"]} for row in keys[start : start + 1000]
                    ],
                    "Quiet": True,
                },
            )
            if response.get("Errors"):
                raise RuntimeError(f"R2 deletion failed: {response['Errors']}")
        client.put_object(
            Bucket=bucket,
            Key=index_key,
            Body=(
                json.dumps([row for row in index if row["path"] != path], indent=2)
                + "\n"
            ).encode("ascii"),
            ContentType="application/json",
            CacheControl="no-cache",
        )
        print(f"R2 directory deleted: {path} {entry['size']} bytes", flush=True)
    client.delete_object(Bucket=bucket, Key=journal_key)


def main() -> None:

    from publication import object_store

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("retention.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scope", choices=("engine", "base", "tools"), default="base")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--old-naming", action="store_true")
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    policy = json.loads(args.config.read_text(encoding="ascii"))
    client, bucket = object_store(), os.environ["R2_BUCKET"]
    with publication_lock(client, bucket):
        if args.command:
            subprocess.run(args.command, check=True)
            return
        report = plan(client, bucket, policy, old_naming=args.old_naming)
        data = json.dumps(report, indent=2) + "\n"
        print(data, end="", flush=True)
        if args.output:
            args.output.write_text(data, encoding="ascii")
        approval_fields = (
            ("toolsDeletionApproval", "toolsDryRunListing")
            if args.scope == "tools"
            else ("deletionApproval", "dryRunListing")
        )
        if (
            all(policy.get(field) is not None for field in approval_fields)
            and not report["unresolvedConsumerPins"]
            and (args.scope != "engine" or report["engineDeletionEnabled"])
            and (args.scope != "tools" or report["toolsDeletionEnabled"])
            and not args.dry_run
        ):
            apply(client, bucket, policy, args.scope, report)
            after = plan(client, bucket, policy, old_naming=args.old_naming)
            print(
                "Retention after application: "
                + json.dumps(
                    {"bucketBytes": after["bucketBytes"], "channels": after["channels"]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
