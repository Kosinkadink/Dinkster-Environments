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
from pathlib import Path

from botocore.exceptions import ClientError
from packaging.version import Version


def select_publications(
    entries: list[dict],
    channels: list[dict],
    *,
    engine_count: int,
    predecessors: int,
    verified_aliases: frozenset[str] = frozenset(),
) -> dict:
    """Keep physical channel paths, stable engines and distinct base predecessors."""
    if type(engine_count) is not int or engine_count < 0:
        raise ValueError("engineMainReleases must be a nonnegative integer")
    if type(predecessors) is not int or predecessors < 0:
        raise ValueError("basePredecessors must be a nonnegative integer")
    by_path = {entry["path"]: entry for entry in entries}
    if len(by_path) != len(entries):
        raise ValueError("Duplicate publication path")
    keep = set()
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
        if key.startswith("channels/") or key.endswith(("/extras.json", "/index.json")):
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
                "size": sum(
                    row["Size"]
                    for name, row in objects.items()
                    if name.startswith(prefix + "/")
                ),
            }
        )
    return result


def plan(client, bucket: str, policy: dict) -> dict:
    objects, metadata = inventory(client, bucket)
    releases = {
        repo: github_releases(repo)
        for repo in ("Kosinkadink/Dinkster", "Kosinkadink/Dinkster-Environments")
    }
    entries = publications(objects, metadata, releases)
    for scope in ("engine", "base"):
        journal = read_optional(client, bucket, f"retention/{scope}-pending.json")
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
    selection = select_publications(
        entries,
        channels,
        engine_count=policy["engineMainReleases"],
        predecessors=policy["basePredecessors"],
        verified_aliases=frozenset(policy["verifiedAliasPaths"]),
    )
    keep = set(selection["keep"])
    return {
        "mode": "dry-run",
        "bucketBytes": sum(row["Size"] for row in objects.values()),
        "delete": [entry for entry in entries if entry["path"] not in keep],
        "keep": sorted(keep),
        "baseVariants": selection["baseVariants"],
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
                if key.startswith(("engine/", "base/"))
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
    for field in ("deletionApproval", "dryRunListing"):
        if not re.fullmatch(
            r"https://github\.com/Kosinkadink/comfy-vibe-station/issues/623#issuecomment-[0-9]+",
            policy.get(field) or "",
        ):
            raise ValueError(f"Deletion requires the recorded {field}")
    repository = "Kosinkadink/" + (
        "Dinkster" if scope == "engine" else "Dinkster-Environments"
    )
    if (
        os.environ.get("GITHUB_REPOSITORY") != repository
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
    ):
        raise ValueError(
            "Retention application is restricted to the owning repository main"
        )
    client.put_object(
        Bucket=bucket,
        Key="retention/base-publication-order.json",
        Body=(json.dumps(report["basePublicationOrder"]) + "\n").encode("ascii"),
        ContentType="application/json",
    )
    journal_key = f"retention/{scope}-pending.json"
    journal = read_optional(client, bucket, journal_key)
    objects, _ = inventory(client, bucket)
    if journal is None:
        entries = [
            entry for entry in report["delete"] if entry["repository"] == repository
        ]
        for entry in entries:
            entry["objects"] = [
                {"Key": key, "Size": row["Size"], "ETag": row["ETag"]}
                for key, row in objects.items()
                if key.startswith(entry["path"] + "/")
            ]
        journal = {
            "approval": policy["deletionApproval"],
            "dryRunListing": policy["dryRunListing"],
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
    retained_tags = {
        publication["tag"]
        for publication in report["publications"]
        if publication["path"] in report["keep"]
    }
    for entry in journal["delete"]:
        path, tag = entry["path"], entry["tag"]
        if (
            entry["repository"] != repository
            or path in protected
            or path in report["keep"]
            or tag in retained_tags
            or not path.startswith(scope + "/")
            or not re.fullmatch(r"[A-Za-z0-9._+/-]+", path)
            or ".." in path.split("/")
            or not tag.startswith("main-" if scope == "engine" else "base-")
        ):
            raise ValueError(f"Pending deletion is no longer safe: {path}")
        keys = entry["objects"]
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
            "engine" if scope == "engine" else "base/" + entry["variant"]
        ) + "/index.json"
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
                or (scope == "engine" and not release["prerelease"])
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
    parser.add_argument("--scope", choices=("engine", "base"), default="base")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    policy = json.loads(args.config.read_text(encoding="ascii"))
    client, bucket = object_store(), os.environ["R2_BUCKET"]
    with publication_lock(client, bucket):
        if args.command:
            subprocess.run(args.command, check=True)
            return
        report = plan(client, bucket, policy)
        data = json.dumps(report, indent=2) + "\n"
        print(data, end="", flush=True)
        if args.output:
            args.output.write_text(data, encoding="ascii")
        if (
            policy["deletionApproval"] is not None
            and policy["dryRunListing"] is not None
            and not args.dry_run
        ):
            apply(client, bucket, policy, args.scope, report)
            after = plan(client, bucket, policy)
            print(
                "Retention after application: "
                + json.dumps(
                    {"bucketBytes": after["bucketBytes"], "channels": after["channels"]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
