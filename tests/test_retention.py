from __future__ import annotations

import hashlib
import io
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

import retention
from retention import apply, plan, publication_lock, select_publications


@pytest.fixture
def publications() -> list[dict]:
    engines = [
        {
            "path": f"engine/build-{number}",
            "identity": f"sha256:{number:064x}",
            "tag": f"main-build-{number}",
            "variant": None,
            "label": None,
            "version": f"1.0.0.dev{20 - number}",
            "publishedAt": f"2026-10-{number + 1:02d}T00:00:00Z",
            "complete": True,
            "thirdPartyWheels": False,
        }
        for number in range(6)
    ]
    engines[1]["tag"] = "v1.0.0"
    engines[4]["thirdPartyWheels"] = True
    bases = [
        {
            "path": f"base/{variant}/{number:064x}",
            "identity": f"sha256:{number:064x}",
            "tag": f"base-{variant}-{number:064x}",
            "variant": variant,
            "label": None,
            "version": None,
            "publishedAt": f"2026-10-{number + 1:02d}T00:00:00Z",
            "publicationOrder": number,
            "complete": True,
            "thirdPartyWheels": False,
        }
        for variant in ("linux-cpu", "win-cpu")
        for number in range(4)
    ]
    return engines + bases


@pytest.fixture
def channel(publications: list[dict]) -> dict:
    def reference(entry: dict, field: str) -> dict:
        return {
            field: entry["identity"],
            "locations": [
                {
                    "type": "mirror",
                    "path": entry["path"]
                    + (
                        "/release-manifest.json"
                        if field == "release"
                        else "/manifest.json"
                    ),
                }
            ],
        }

    return {
        "format": "dinkster.channel/1",
        "engine": reference(publications[0], "release"),
        "base": {
            variant: reference(
                next(
                    entry
                    for entry in publications
                    if entry["variant"] == variant
                    and entry["identity"] == f"sha256:{3:064x}"
                ),
                "id",
            )
            for variant in ("linux-cpu", "win-cpu")
        },
    }


def test_keep_set_protects_channel_stable_and_newest_without_bundled_dependencies(
    publications: list[dict],
    channel: dict,
) -> None:
    selection = select_publications(
        publications, [channel], engine_count=2, predecessors=1
    )
    assert set(selection["keep"]) == {
        "engine/build-0",
        "engine/build-1",
        "engine/build-5",
        *(
            f"base/{variant}/{number:064x}"
            for variant in ("linux-cpu", "win-cpu")
            for number in (2, 3)
        ),
    }
    channel["engine"]["release"] = publications[4]["identity"]
    channel["engine"]["locations"][0]["path"] = (
        publications[4]["path"] + "/release-manifest.json"
    )
    assert (
        "engine/build-4"
        in select_publications(publications, [channel], engine_count=0, predecessors=0)[
            "keep"
        ]
    )


def test_labelled_alias_never_counts_as_a_predecessor_or_removes_a_channel_path(
    publications: list[dict],
    channel: dict,
) -> None:
    for number in (2, 3):
        original = next(
            entry
            for entry in publications
            if entry["path"] == f"base/linux-cpu/{number:064x}"
        )
        alias = {
            **original,
            "label": f"label-{number}",
            "path": f"base/linux-cpu/label-{number}",
            "tag": f"base-linux-cpu-label-{number}",
            "publishedAt": f"2026-11-{6 - number:02d}T00:00:00Z",
            "publicationOrder": 6 - number,
        }
        publications.append(alias)
    selection = select_publications(
        publications, [channel], engine_count=0, predecessors=1
    )
    kept = selection["keep"]
    assert f"base/linux-cpu/{3:064x}" in kept
    assert "base/linux-cpu/label-3" in kept
    assert "base/linux-cpu/label-2" in kept
    assert f"base/linux-cpu/{2:064x}" in kept
    channel["base"]["linux-cpu"]["locations"][0]["path"] = (
        "base/linux-cpu/label-3/manifest.json"
    )
    kept = select_publications(publications, [channel], engine_count=0, predecessors=1)[
        "keep"
    ]
    assert f"base/linux-cpu/{3:064x}" in kept
    selection = select_publications(
        publications,
        [channel],
        engine_count=0,
        predecessors=1,
        verified_aliases=frozenset({"base/linux-cpu/label-3"}),
    )
    kept = selection["keep"]
    assert f"base/linux-cpu/{3:064x}" not in kept
    assert "base/linux-cpu/label-2" in kept
    details = selection["baseVariants"]["linux-cpu"]
    assert details["targets"] == ["base/linux-cpu/label-3"]
    assert details["previous"] == ["base/linux-cpu/label-2"]
    assert details["pending"] == []
    assert details["aliases"] == [
        {
            "identity": f"sha256:{number:064x}",
            "paths": [
                f"base/linux-cpu/{number:064x}",
                f"base/linux-cpu/label-{number}",
            ],
            "publicationOrder": number,
            "retainedPaths": (
                [f"base/linux-cpu/{number:064x}", f"base/linux-cpu/label-{number}"]
                if number == 2
                else ["base/linux-cpu/label-3"]
            ),
            "deletablePaths": [f"base/linux-cpu/{3:064x}"] if number == 3 else [],
        }
        for number in (2, 3)
    ]
    selection = select_publications(
        publications,
        [channel],
        engine_count=0,
        predecessors=1,
        verified_aliases=frozenset(
            {"base/linux-cpu/label-2", "base/linux-cpu/label-3"}
        ),
    )
    assert f"base/linux-cpu/{2:064x}" not in selection["keep"]
    assert "base/linux-cpu/label-2" in selection["keep"]


def test_pending_bases_survive_until_channel_advances(
    publications: list[dict],
    channel: dict,
) -> None:
    reference = channel["base"]["linux-cpu"]
    reference["id"] = f"sha256:{1:064x}"
    reference["locations"][0]["path"] = f"base/linux-cpu/{1:064x}/manifest.json"
    for entry in publications:
        if entry["variant"] is not None:
            entry["publishedAt"] = "2026-11-01T00:00:00Z"
    selection = select_publications(
        list(reversed(publications)),
        [channel],
        engine_count=0,
        predecessors=1,
    )
    assert selection["baseVariants"]["linux-cpu"] == {
        "targets": [f"base/linux-cpu/{1:064x}"],
        "previous": [f"base/linux-cpu/{0:064x}"],
        "pending": [f"base/linux-cpu/{2:064x}", f"base/linux-cpu/{3:064x}"],
        "aliases": [],
    }
    assert all(
        f"base/linux-cpu/{number:064x}" in selection["keep"] for number in range(4)
    )
    reference["id"] = f"sha256:{3:064x}"
    reference["locations"][0]["path"] = f"base/linux-cpu/{3:064x}/manifest.json"
    selection = select_publications(
        publications, [channel], engine_count=0, predecessors=1
    )
    assert selection["baseVariants"]["linux-cpu"]["previous"] == [
        f"base/linux-cpu/{2:064x}"
    ]
    assert selection["baseVariants"]["linux-cpu"]["pending"] == []
    assert f"base/linux-cpu/{0:064x}" not in selection["keep"]
    assert f"base/linux-cpu/{1:064x}" not in selection["keep"]


def test_shared_engine_tag_cannot_remove_a_retained_publication(
    publications: list[dict], channel: dict
) -> None:
    publications[2]["tag"] = publications[5]["tag"]
    selection = select_publications(
        publications, [channel], engine_count=1, predecessors=1
    )
    assert "engine/build-2" in selection["keep"]
    assert selection["sharedTags"] == {
        "main-build-5": ["engine/build-2", "engine/build-5"]
    }
    publications[4]["tag"] = publications[5]["tag"]
    selection = select_publications(
        publications, [channel], engine_count=2, predecessors=1
    )
    assert "engine/build-3" in selection["keep"]
    assert "engine/build-4" in selection["keep"]
    assert "engine/build-5" in selection["keep"]


@pytest.mark.parametrize("damage", ["absent", "identity", "incomplete", "no-channels"])
def test_invalid_channel_refuses_retention(
    publications: list[dict],
    channel: dict,
    damage: str,
) -> None:
    channel = deepcopy(channel)
    if damage == "absent":
        channel["base"]["win-cpu"]["locations"][0]["path"] = (
            "base/win-cpu/absent/manifest.json"
        )
    elif damage == "identity":
        channel["engine"]["release"] = "sha256:" + "f" * 64
    elif damage == "incomplete":
        publications[0]["complete"] = False
    with pytest.raises(ValueError, match="Channel target|No channels"):
        select_publications(
            publications,
            [] if damage == "no-channels" else [channel],
            engine_count=10,
            predecessors=1,
        )


class MemoryBucket:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.fail_delete = False
        self.fail_put: str | None = None
        self.writes = 0
        self.listed_bytes = 0

    def get_paginator(self, operation):
        return self

    def paginate(self, **kwargs):
        self.listed_bytes = sum(len(data) for data in self.data.values())
        rows = [
            {
                "Key": key,
                "Size": len(data),
                "ETag": hashlib.sha256(data).hexdigest(),
                "LastModified": datetime(2026, 11, 3, tzinfo=UTC),
            }
            for key, data in self.data.items()
        ]
        return [{"Contents": rows[:2]}, {"Contents": rows[2:]}]

    def get_object(self, *, Key, **kwargs):
        if Key not in self.data:
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": 404}}, "GetObject"
            )
        return {
            "Body": io.BytesIO(self.data[Key]),
            "ETag": hashlib.sha256(self.data[Key]).hexdigest(),
        }

    def put_object(self, *, Key, Body, IfMatch=None, IfNoneMatch=None, **kwargs):
        if Key == self.fail_put:
            self.fail_put = None
            raise RuntimeError("Index write failed")
        existing = self.data.get(Key)
        if (IfNoneMatch == "*" and existing is not None) or (
            IfMatch is not None
            and (existing is None or hashlib.sha256(existing).hexdigest() != IfMatch)
        ):
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": 412}}, "PutObject"
            )
        self.writes += 1
        self.data[Key] = Body
        return {"ETag": hashlib.sha256(Body).hexdigest()}

    def delete_objects(self, *, Delete, **kwargs):
        for row in Delete["Objects"]:
            self.delete_object(Key=row["Key"])
            if self.fail_delete:
                self.fail_delete = False
                return {"Errors": [{"Code": "InternalError"}]}
        return {}

    def delete_object(self, *, Key, **kwargs):
        self.writes += 1
        self.data.pop(Key, None)


@pytest.fixture
def retention_store(monkeypatch):
    bucket = MemoryBucket()
    monkeypatch.setattr(retention, "consumer_pins", lambda: {"sources": {}, "pins": []})
    paths = []
    for version, commit in (("1.0.0.dev1", "a" * 40), ("1.0.0", "b" * 40)):
        manifest = {
            "repository": "Kosinkadink/Dinkster",
            "version": version,
            "commit": commit,
            "tag": "v1.0.0" if version == "1.0.0" else "main-" + commit,
            "workspaceWheels": ["engine.whl"],
            "artifacts": [{"name": "engine.whl", "size": 7}],
        }
        data = json.dumps(manifest).encode()
        path = "engine/" + hashlib.sha256(data).hexdigest()
        bucket.data[path + "/release-manifest.json"] = data
        bucket.data[path + "/engine.whl"] = b"payload"
        bucket.data[path + "/SHA256SUMS"] = b"checksums"
        paths.append(path)
    channel = {
        "format": "dinkster.channel/1",
        "engine": {
            "release": "sha256:" + paths[1].split("/")[1],
            "locations": [
                {"type": "mirror", "path": paths[1] + "/release-manifest.json"}
            ],
        },
        "base": {},
    }
    bucket.data["channels/main.json"] = json.dumps(channel).encode()
    bucket.data["engine/index.json"] = json.dumps(
        [{"path": path} for path in reversed(paths)]
    ).encode()
    bucket.data["engine/unknown/file.bin"] = b"unclassified"
    tag = "main-" + "a" * 40
    release = {"id": 7, "tag_name": tag, "draft": False, "prerelease": True}
    github = {"release": release, "tag": {"type": "commit", "sha": "a" * 40}}

    def releases(repository):
        return [github["release"]] if github["release"] is not None else []

    def read_api(args, **kwargs):
        if "/git/tags/" in args[-1]:
            return json.dumps({"object": {"type": "commit", "sha": "a" * 40}})
        return json.dumps(
            [{"ref": "refs/tags/" + tag, "object": github["tag"]}]
            if github["tag"] is not None
            else []
        )

    def delete_api(args, **kwargs):
        if "/releases/" in args[-1]:
            github["release"] = None
        else:
            github["tag"] = None

    monkeypatch.setattr(retention, "github_releases", releases)
    monkeypatch.setattr(retention.subprocess, "check_output", read_api)
    monkeypatch.setattr(retention.subprocess, "run", delete_api)
    monkeypatch.setenv("GITHUB_REPOSITORY", "Kosinkadink/Dinkster")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    policy = {
        "engineMainReleases": 0,
        "engineDeletionEnabled": True,
        "basePredecessors": 1,
        "verifiedAliasPaths": [],
        "deletionApproval": "https://github.com/Kosinkadink/comfy-vibe-station/issues/623#issuecomment-1",
        "dryRunListing": "https://github.com/Kosinkadink/comfy-vibe-station/issues/623#issuecomment-2",
    }
    return bucket, policy, paths, github


def test_authenticated_plan_is_read_only_and_lists_unknown_directories(retention_store):
    bucket, policy, paths, _ = retention_store
    before = deepcopy(bucket.data)
    report = plan(bucket, "bucket", policy)
    assert bucket.data == before
    assert bucket.writes == 0
    assert [row["path"] for row in report["delete"]] == [paths[0]]
    assert report["keep"] == [paths[1]]
    assert report["unclassifiedDirectories"] == ["engine/unknown"]
    assert report["reclaimedBytes"] == sum(
        len(data) for key, data in before.items() if key.startswith(paths[0] + "/")
    )


def test_old_consumer_pin_retains_native_payload_and_release_beyond_newest_ten(
    retention_store, monkeypatch
):
    bucket, policy, paths, github = retention_store
    policy["engineMainReleases"] = 10
    releases = []
    for number in range(13):
        manifest = {
            "repository": "Kosinkadink/Dinkster",
            "version": f"1.0.0.dev{number + 1}",
            "commit": f"{number + 1:040x}",
            "tag": f"main-{number + 1:040x}",
            "workspaceWheels": ["engine.whl"],
            "artifacts": [
                {"name": name, "size": len(name.encode())}
                for name in (
                    "engine.whl",
                    "dinkster-tools-windows-x86_64.zip",
                    "dinkster-tools-linux-x86_64.zip",
                    "dinkster-tools-macos-arm64.zip",
                )
            ],
        }
        data = json.dumps(manifest).encode()
        path = "engine/" + hashlib.sha256(data).hexdigest()
        bucket.data[path + "/release-manifest.json"] = data
        bucket.data[path + "/SHA256SUMS"] = b"checksums"
        for row in manifest["artifacts"]:
            bucket.data[path + "/" + row["name"]] = row["name"].encode()
        releases.append(
            {
                "id": number + 100,
                "tag_name": manifest["tag"],
                "draft": False,
                "prerelease": True,
                "published_at": f"2026-10-{number + 1:02d}T00:00:00Z",
                "path": path,
            }
        )
    monkeypatch.setattr(
        retention,
        "github_releases",
        lambda repository: (
            releases + ([github["release"]] if github["release"] else [])
        ),
    )
    pinned = releases[0]
    consumers = {
        "sources": {"desktop": "current"},
        "pins": [
            {
                "consumer": "Dinkster-Desktop",
                "repository": "Kosinkadink/Dinkster",
                "tag": pinned["tag_name"],
                "identity": "sha256:" + pinned["path"].split("/")[1],
            }
        ],
    }
    monkeypatch.setattr(retention, "consumer_pins", lambda: consumers)
    report = plan(bucket, "bucket", policy)
    assert pinned["path"] in report["keep"]
    assert releases[1]["path"] in {entry["path"] for entry in report["delete"]}
    before = {
        key: data
        for key, data in bucket.data.items()
        if key.startswith(pinned["path"] + "/")
    }
    control = releases[1]
    tags = {release["tag_name"] for release in releases} | {
        github["release"]["tag_name"]
    }
    bucket.data["engine/index.json"] = json.dumps(
        [{"path": entry["path"]} for entry in report["publications"]]
    ).encode()

    def api(args, **kwargs):
        tag = args[-1].rsplit("/", 1)[-1]
        return json.dumps(
            [{"ref": "refs/tags/" + tag, "object": {"type": "commit", "sha": tag[5:]}}]
        )

    def delete(args, **kwargs):
        if "/releases/" in args[-1]:
            releases[:] = [
                r for r in releases if str(r["id"]) != args[-1].rsplit("/", 1)[-1]
            ]
            if args[-1].endswith("/7"):
                github["release"] = None
        else:
            tags.remove(args[-1].rsplit("/", 1)[-1])

    monkeypatch.setattr(retention.subprocess, "check_output", api)
    monkeypatch.setattr(retention.subprocess, "run", delete)
    apply(bucket, "bucket", policy, "engine", report)
    assert all(bucket.data[key] == data for key, data in before.items())
    assert len(before) == 6
    assert pinned in releases and pinned["tag_name"] in tags
    assert not any(key.startswith(control["path"] + "/") for key in bucket.data)
    assert control["tag_name"] not in tags
    assert control not in releases
    assert pinned["path"] in {
        row["path"] for row in json.loads(bucket.data["engine/index.json"])
    }
    assert paths[1] in report["keep"]


@pytest.mark.parametrize("when", ["plan", "apply", "changed"])
def test_consumer_read_failure_or_change_prevents_deletion(
    retention_store, monkeypatch, when
):
    bucket, policy, _, github = retention_store
    report = plan(bucket, "bucket", policy)
    before = deepcopy(bucket.data)

    def pins():
        if when == "changed":
            return {"sources": {"desktop": "changed"}, "pins": []}
        raise RuntimeError("Consumer source unreadable")

    monkeypatch.setattr(retention, "consumer_pins", pins)
    with pytest.raises((RuntimeError, ValueError), match="[Cc]onsumer"):
        if when == "plan":
            plan(bucket, "bucket", policy)
        else:
            apply(bucket, "bucket", policy, "engine", report)
    assert bucket.data == before
    assert github["release"] is not None and github["tag"] is not None


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "format",
        "tag",
        "digest",
        "bases",
        "missing",
        "fixture-commit",
        "fixture-digest",
        "fixture-source",
        "fixture-engine",
        "fixture-base",
    ],
)
def test_public_pin_sources_are_required_and_validated(monkeypatch, invalid):
    documents = {
        "Dinkster-Desktop": {
            "repository": "Kosinkadink/Dinkster",
            "releaseTag": "main-tools",
            "manifest": {"sha256": "a" * 64},
        },
        "Dinkster-Environments": {
            "format": "dinkster.consumer-pins/1",
            "consumers": {
                "dinkster-registry": {
                    "repository": "Kosinkadink/Dinkster",
                    "releaseTag": "main-packs",
                }
            },
            "nonReleaseConsumers": {
                "registry-fixtures": {
                    "repository": "Kosinkadink/dinkster-registry-service",
                    "commit": "c" * 40,
                    "path": "scripts/release-packs.lock",
                    "sha256": "d" * 64,
                    "source": "registry:https://registry.dinkster.org",
                    "engineReleaseDependency": False,
                    "baseReleaseDependency": False,
                }
            },
        },
        "Dinkster": {"linux-cpu": "b" * 64},
    }
    if invalid == "format":
        documents["Dinkster-Environments"]["format"] = "unknown"
    elif invalid == "tag":
        documents["Dinkster-Desktop"]["releaseTag"] = "../wrong"
    elif invalid == "digest":
        documents["Dinkster-Desktop"]["manifest"]["sha256"] = "short"
    elif invalid == "bases":
        documents["Dinkster"] = {}
    elif invalid == "missing":
        del documents["Dinkster-Desktop"]["releaseTag"]
    elif invalid and invalid.startswith("fixture-"):
        field, value = {
            "fixture-commit": ("commit", "short"),
            "fixture-digest": ("sha256", "invalid"),
            "fixture-source": ("source", "https://example.com/engine.zip"),
            "fixture-engine": ("engineReleaseDependency", True),
            "fixture-base": ("baseReleaseDependency", "false"),
        }[invalid]
        documents["Dinkster-Environments"]["nonReleaseConsumers"]["registry-fixtures"][
            field
        ] = value

    def read(args, **kwargs):
        repository = args[-1].split("/")[2]
        return json.dumps(documents[repository]).encode()

    monkeypatch.setattr(retention.subprocess, "check_output", read)
    if invalid:
        with pytest.raises((ValueError, KeyError)):
            retention.consumer_pins()
    else:
        result = retention.consumer_pins()
        assert len(result["sources"]) == 3
        assert result["nonReleaseConsumers"] == {
            "registry-fixtures": {
                "repository": "Kosinkadink/dinkster-registry-service",
                "commit": "c" * 40,
                "path": "scripts/release-packs.lock",
                "sha256": "d" * 64,
                "source": "registry:https://registry.dinkster.org",
                "engineReleaseDependency": False,
                "baseReleaseDependency": False,
            }
        }
        assert result["pins"] == [
            {
                "consumer": "Dinkster-Desktop",
                "repository": "Kosinkadink/Dinkster",
                "tag": "main-tools",
                "identity": "sha256:" + "a" * 64,
                "status": "active",
            },
            {
                "consumer": "dinkster-registry",
                "repository": "Kosinkadink/Dinkster",
                "tag": "main-packs",
                "status": "active",
            },
            {
                "consumer": "Dinkster",
                "repository": "Kosinkadink/Dinkster-Environments",
                "variant": "linux-cpu",
                "identity": "sha256:" + "b" * 64,
            },
        ]


def test_new_consumer_pin_blocks_a_preexisting_deletion_journal(
    retention_store, monkeypatch
):
    bucket, policy, paths, github = retention_store
    report = plan(bucket, "bucket", policy)
    pending = report["delete"][0]
    bucket.data["retention/engine-pending.json"] = json.dumps(
        {
            "delete": [
                {
                    **pending,
                    "objects": [
                        {
                            "Key": key,
                            "Size": len(data),
                            "ETag": hashlib.sha256(data).hexdigest(),
                        }
                        for key, data in bucket.data.items()
                        if key.startswith(paths[0] + "/")
                    ],
                }
            ]
        }
    ).encode()
    consumers = {
        "sources": {"desktop": "new"},
        "pins": [
            {
                "consumer": "Desktop",
                "repository": "Kosinkadink/Dinkster",
                "tag": pending["tag"],
            }
        ],
    }
    monkeypatch.setattr(retention, "consumer_pins", lambda: consumers)
    report = plan(bucket, "bucket", policy)
    assert paths[0] in report["keep"] and report["delete"] == []
    before = {
        key: data for key, data in bucket.data.items() if key.startswith(paths[0] + "/")
    }
    with pytest.raises(ValueError, match="no longer safe"):
        apply(bucket, "bucket", policy, "engine", report)
    assert all(bucket.data[key] == data for key, data in before.items())
    assert github["release"] is not None and github["tag"] is not None


@pytest.mark.parametrize("missing", ["deletionApproval", "dryRunListing"])
def test_apply_needs_both_recorded_gates(retention_store, missing):
    bucket, policy, _, github = retention_store
    report = plan(bucket, "bucket", policy)
    before = deepcopy(bucket.data)
    policy[missing] = None
    with pytest.raises(ValueError, match=missing):
        apply(bucket, "bucket", policy, "engine", report)
    assert bucket.data == before
    assert bucket.writes == 0
    assert github["release"] is not None


@pytest.mark.parametrize("gate", ["deletionApproval", "dryRunListing", "--dry-run"])
def test_cli_emits_dry_run_without_deleting_before_all_gates(
    retention_store, gate, monkeypatch, tmp_path
):
    bucket, policy, paths, github = retention_store
    if gate != "--dry-run":
        policy[gate] = None
    config = tmp_path / "retention.json"
    config.write_text(json.dumps(policy))
    output = tmp_path / "report.json"
    args = ["retention.py", "--config", str(config), "--output", str(output)]
    if gate == "--dry-run":
        args.append(gate)
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr("publication.object_store", lambda: bucket)
    monkeypatch.setenv("R2_BUCKET", "bucket")
    before = deepcopy(bucket.data)
    retention.main()
    report = json.loads(output.read_bytes())
    assert report["mode"] == "dry-run"
    assert [row["path"] for row in report["delete"]] == [paths[0]]
    assert all(bucket.data[key] == value for key, value in before.items())
    assert github["release"] is not None
    assert github["tag"] is not None
    assert "retention/base-pending.json" not in bucket.data
    assert json.loads(bucket.data["retention/publication-lock.json"])["owner"] is None


def test_cli_lock_wrapper_does_not_run_a_second_retention_step(
    retention_store, monkeypatch, tmp_path
):
    bucket, policy, _, github = retention_store
    config = tmp_path / "retention.json"
    config.write_text(json.dumps(policy))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retention.py",
            "--config",
            str(config),
            "--output",
            str(output),
            "--scope",
            "engine",
            "--command",
            "publisher",
        ],
    )
    monkeypatch.setattr("publication.object_store", lambda: bucket)
    monkeypatch.setenv("R2_BUCKET", "bucket")

    def publisher(args, **kwargs):
        assert json.loads(bucket.data["retention/publication-lock.json"])["owner"]
        bucket.data["published"] = b"new publication"

    monkeypatch.setattr(retention.subprocess, "run", publisher)
    before = deepcopy(bucket.data)
    retention.main()
    assert bucket.data["published"] == b"new publication"
    assert all(bucket.data[key] == value for key, value in before.items())
    assert not output.exists()
    assert github["release"] is not None
    assert github["tag"] is not None


def test_cli_application_reports_final_inventory_and_preserved_channel(
    retention_store, monkeypatch, tmp_path, capsys
):
    bucket, policy, paths, github = retention_store
    config = tmp_path / "retention.json"
    config.write_text(json.dumps(policy))
    monkeypatch.setattr(
        sys, "argv", ["retention.py", "--config", str(config), "--scope", "engine"]
    )
    monkeypatch.setattr("publication.object_store", lambda: bucket)
    monkeypatch.setenv("R2_BUCKET", "bucket")
    channel = bucket.data["channels/main.json"]
    retention.main()
    summary = json.loads(
        capsys.readouterr().out.split("Retention after application: ")[1]
    )
    assert summary["bucketBytes"] == bucket.listed_bytes
    assert summary["channels"] == {"channels/main.json": json.loads(channel)}
    assert bucket.data["channels/main.json"] == channel
    assert paths[1] + "/engine.whl" in bucket.data
    assert not any(key.startswith(paths[0] + "/") for key in bucket.data)
    assert github["release"] is None
    assert github["tag"] is None


@pytest.mark.parametrize(
    "tag_kind", ["lightweight", "annotated", "unmatched", "absent"]
)
def test_partial_deletion_resumes_to_matching_release_and_retained_index(
    retention_store, tag_kind, capsys
):
    bucket, policy, paths, github = retention_store
    if tag_kind == "annotated":
        github["tag"] = {"type": "tag", "sha": "c" * 40}
    elif tag_kind == "unmatched":
        github["tag"] = {"type": "commit", "sha": "d" * 40}
    elif tag_kind == "absent":
        github["release"] = None
        github["tag"] = None
    original_tag = deepcopy(github["tag"])
    retained = {
        key: data
        for key, data in bucket.data.items()
        if key.startswith(paths[1] + "/") or key.startswith("channels/")
    }
    report = plan(bucket, "bucket", policy)
    bucket.fail_delete = True
    with pytest.raises(RuntimeError, match="R2 deletion failed"):
        apply(bucket, "bucket", policy, "engine", report)
    assert "retention/engine-pending.json" in bucket.data
    assert github["release"] is None
    # The first deletion removed the old manifest, not all its payloads.
    assert paths[0] + "/release-manifest.json" not in bucket.data
    assert paths[0] + "/engine.whl" in bucket.data
    apply(bucket, "bucket", policy, "engine", plan(bucket, "bucket", policy))
    assert not any(key.startswith(paths[0] + "/") for key in bucket.data)
    assert "retention/engine-pending.json" not in bucket.data
    assert json.loads(bucket.data["engine/index.json"]) == [{"path": paths[1]}]
    assert all(bucket.data[key] == data for key, data in retained.items())
    assert bucket.data["engine/unknown/file.bin"] == b"unclassified"
    assert github["tag"] == (original_tag if tag_kind == "unmatched" else None)
    if tag_kind == "unmatched":
        assert "GitHub tag unmatched, retained" in capsys.readouterr().out


def test_resume_refuses_changed_payload(retention_store):
    bucket, policy, paths, _ = retention_store
    bucket.fail_delete = True
    with pytest.raises(RuntimeError, match="R2 deletion failed"):
        apply(bucket, "bucket", policy, "engine", plan(bucket, "bucket", policy))
    bucket.data[paths[0] + "/engine.whl"] = b"changed"
    before = deepcopy(bucket.data)
    with pytest.raises(ValueError, match="changed since planning"):
        plan(bucket, "bucket", policy)
    assert bucket.data == before


def test_multiple_directories_resume_after_payload_delete_and_index_failure(
    retention_store,
):
    bucket, policy, paths, _ = retention_store
    manifest = json.loads(bucket.data[paths[0] + "/release-manifest.json"])
    manifest.update(commit="e" * 40, tag="main-" + "e" * 40)
    data = json.dumps(manifest).encode()
    extra = "engine/" + hashlib.sha256(data).hexdigest()
    bucket.data[extra + "/release-manifest.json"] = data
    bucket.data[extra + "/engine.whl"] = b"payload"
    bucket.data[extra + "/SHA256SUMS"] = b"checksums"
    bucket.data["engine/index.json"] = json.dumps(
        [{"path": path} for path in [paths[1], extra, paths[0]]]
    ).encode()
    report = plan(bucket, "bucket", policy)
    assert {row["path"] for row in report["delete"]} == {paths[0], extra}
    bucket.fail_put = "engine/index.json"
    with pytest.raises(RuntimeError, match="Index write failed"):
        apply(bucket, "bucket", policy, "engine", report)
    assert not any(key.startswith(paths[0] + "/") for key in bucket.data)
    assert extra + "/engine.whl" in bucket.data
    assert "retention/engine-pending.json" in bucket.data
    apply(bucket, "bucket", policy, "engine", plan(bucket, "bucket", policy))
    assert not any(key.startswith(extra + "/") for key in bucket.data)
    assert json.loads(bucket.data["engine/index.json"]) == [{"path": paths[1]}]
    assert "retention/engine-pending.json" not in bucket.data


def test_resume_stops_if_channel_moves_to_a_pending_directory(retention_store):
    bucket, policy, paths, _ = retention_store
    bucket.fail_put = "engine/index.json"
    with pytest.raises(RuntimeError, match="Index write failed"):
        apply(bucket, "bucket", policy, "engine", plan(bucket, "bucket", policy))
    channel = json.loads(bucket.data["channels/main.json"])
    channel["engine"]["release"] = "sha256:" + paths[0].split("/")[1]
    channel["engine"]["locations"][0]["path"] = paths[0] + "/release-manifest.json"
    bucket.data["channels/main.json"] = json.dumps(channel).encode()
    before = deepcopy(bucket.data)
    with pytest.raises(ValueError, match="Channel target is missing or incomplete"):
        plan(bucket, "bucket", policy)
    assert bucket.data == before


def test_completed_directory_retry_cannot_remove_a_newly_retained_shared_tag(
    retention_store,
):
    bucket, policy, paths, github = retention_store
    manifest = json.loads(bucket.data[paths[0] + "/release-manifest.json"])
    bucket.fail_put = "engine/index.json"
    with pytest.raises(RuntimeError, match="Index write failed"):
        apply(bucket, "bucket", policy, "engine", plan(bucket, "bucket", policy))
    manifest["builtAt"] = "2026-11-04T00:00:00Z"
    data = json.dumps(manifest).encode()
    path = "engine/" + hashlib.sha256(data).hexdigest()
    bucket.data[path + "/release-manifest.json"] = data
    bucket.data[path + "/engine.whl"] = b"payload"
    bucket.data[path + "/SHA256SUMS"] = b"checksums"
    channel = json.loads(bucket.data["channels/main.json"])
    channel["engine"]["release"] = "sha256:" + path.split("/")[1]
    channel["engine"]["locations"][0]["path"] = path + "/release-manifest.json"
    bucket.data["channels/main.json"] = json.dumps(channel).encode()
    github["release"] = {
        "id": 7,
        "tag_name": manifest["tag"],
        "draft": False,
        "prerelease": True,
    }
    github["tag"] = {"type": "commit", "sha": manifest["commit"]}
    with pytest.raises(ValueError, match="Pending deletion is no longer safe"):
        apply(bucket, "bucket", policy, "engine", plan(bucket, "bucket", policy))
    assert github["release"] is not None
    assert github["tag"] is not None
    assert bucket.data[path + "/engine.whl"] == b"payload"


@pytest.mark.parametrize("damage", [None, "alias-parts", "missing-index"])
def test_base_inventory_orders_aliases_by_index_not_republication_date(
    retention_store, damage
):
    bucket, policy, _, _ = retention_store
    manifest = json.loads(
        (Path(__file__).parent / "fixtures/base/linux-cpu.json").read_bytes()
    )
    manifest["archive"]["parts"] = [
        {"filename": "base.tar.zst.001", "size": 3, "sha256": "a" * 64}
    ]
    identity = manifest["id"]
    original = "base/linux-cpu/" + identity[7:]
    label = "2026-11-03-py3.14.10-torch2.14.1-" + identity[7:15]
    alias = "base/linux-cpu/" + label
    extras = {"format": "dinkster.extras/1", "base": identity, "variant": "linux-cpu"}
    for path in (original, alias):
        document = deepcopy(manifest)
        if path == alias:
            document.update(label=label, builtAt="2026-11-03T00:00:00Z")
            if damage == "alias-parts":
                document["archive"]["parts"][0]["sha256"] = "b" * 64
        bucket.data[path + "/manifest.json"] = json.dumps(document).encode()
        bucket.data[path + "/extras.json"] = json.dumps(extras).encode()
        bucket.data[path + "/base.tar.zst.001"] = b"abc"
    if damage != "missing-index":
        bucket.data["base/linux-cpu/index.json"] = json.dumps(
            [{"path": alias}, {"path": original}]
        ).encode()
    channel = json.loads(bucket.data["channels/main.json"])
    channel["base"]["linux-cpu"] = {
        "id": identity,
        "locations": [{"type": "mirror", "path": alias + "/manifest.json"}],
    }
    bucket.data["channels/main.json"] = json.dumps(channel).encode()
    policy["verifiedAliasPaths"] = [alias]
    before = deepcopy(bucket.data)
    if damage:
        with pytest.raises(
            ValueError, match="different payloads|Missing publication order"
        ):
            plan(bucket, "bucket", policy)
    else:
        report = plan(bucket, "bucket", policy)
        details = report["baseVariants"]["linux-cpu"]
        assert details["targets"] == [alias]
        assert details["pending"] == []
        assert details["previous"] == []
        assert details["aliases"][0]["publicationOrder"] == 1
        assert details["aliases"][0]["deletablePaths"] == [original]
    assert bucket.data == before
    assert bucket.writes == 0


@pytest.fixture
def base_retention_store(retention_store, monkeypatch):
    from bundles import manifest_id

    bucket, policy, _, _ = retention_store
    template = json.loads(
        (Path(__file__).parent / "fixtures/base/linux-cpu.json").read_bytes()
    )
    paths = []
    manifests = []
    for protocol in (11, 12, 13):
        manifest = deepcopy(template)
        manifest["workerProtocol"] = protocol
        manifest["id"] = manifest_id(manifest)
        manifest["archive"]["parts"] = [
            {"filename": "base.tar.zst.001", "size": 3, "sha256": "a" * 64}
        ]
        paths.append("base/linux-cpu/" + manifest["id"][7:])
        manifests.append(manifest)
    label = "2026-11-03-py3.14.10-torch2.14.1-" + manifests[1]["id"][7:15]
    alias = "base/linux-cpu/" + label
    manifests.append({**deepcopy(manifests[1]), "label": label})
    for path, manifest in zip([*paths, alias], manifests, strict=True):
        bucket.data[path + "/manifest.json"] = json.dumps(manifest).encode()
        bucket.data[path + "/extras.json"] = json.dumps(
            {
                "format": "dinkster.extras/1",
                "base": manifest["id"],
                "variant": "linux-cpu",
            }
        ).encode()
        bucket.data[path + "/base.tar.zst.001"] = b"abc"
    bucket.data["base/linux-cpu/index.json"] = json.dumps(
        [{"path": path} for path in [alias, *reversed(paths)]]
    ).encode()
    channel = json.loads(bucket.data["channels/main.json"])
    channel["base"]["linux-cpu"] = {
        "id": manifests[1]["id"],
        "locations": [{"type": "mirror", "path": alias + "/manifest.json"}],
    }
    bucket.data["channels/main.json"] = json.dumps(channel).encode()
    policy["verifiedAliasPaths"] = [alias]
    releases = {
        "base-linux-cpu-" + path.rsplit("/", 1)[1]: {
            "id": number,
            "tag_name": "base-linux-cpu-" + path.rsplit("/", 1)[1],
            "draft": False,
            "prerelease": False,
        }
        for number, path in enumerate([*paths, alias], 101)
    }
    tags = {tag: {"type": "commit", "sha": "f" * 40} for tag in releases}

    def delete_api(args, **kwargs):
        if "/releases/" in args[-1]:
            identifier = int(args[-1].rsplit("/", 1)[1])
            for tag, release in list(releases.items()):
                if release["id"] == identifier:
                    del releases[tag]
        else:
            tags.pop(args[-1].rsplit("/", 1)[1], None)

    def read_api(args, **kwargs):
        tag = args[-1].rsplit("/", 1)[1]
        return json.dumps(
            [{"ref": "refs/tags/" + tag, "object": tags[tag]}] if tag in tags else []
        )

    monkeypatch.setattr(
        retention, "github_releases", lambda repo: list(releases.values())
    )
    monkeypatch.setattr(retention.subprocess, "check_output", read_api)
    monkeypatch.setattr(retention.subprocess, "run", delete_api)
    monkeypatch.setenv("GITHUB_REPOSITORY", "Kosinkadink/Dinkster-Environments")
    return bucket, policy, paths, alias, releases, tags


@pytest.mark.parametrize("failure", [None, "partial-delete", "index-write"])
def test_alias_removal_keeps_generation_order_after_index_rebuild(
    base_retention_store, failure
):
    bucket, policy, paths, alias, releases, tags = base_retention_store
    report = plan(bucket, "bucket", policy)
    details = report["baseVariants"]["linux-cpu"]
    assert details["previous"] == [paths[0]]
    assert details["pending"] == [paths[2]]
    assert [row["path"] for row in report["delete"] if row["variant"]] == [paths[1]]
    if failure:
        bucket.fail_delete = failure == "partial-delete"
        bucket.fail_put = (
            "base/linux-cpu/index.json" if failure == "index-write" else None
        )
        with pytest.raises(RuntimeError, match="R2 deletion failed|Index write failed"):
            apply(bucket, "bucket", policy, "base", report)
        report = plan(bucket, "bucket", policy)
    apply(bucket, "bucket", policy, "base", report)
    assert not any(key.startswith(paths[1] + "/") for key in bucket.data)
    assert "base-linux-cpu-" + paths[1].rsplit("/", 1)[1] not in releases
    assert "base-linux-cpu-" + paths[1].rsplit("/", 1)[1] not in tags
    assert "base-linux-cpu-" + alias.rsplit("/", 1)[1] in releases
    assert "base-linux-cpu-" + alias.rsplit("/", 1)[1] in tags
    # The publisher rebuilds its index from the remaining manifests. The alias
    # has fresh descriptive metadata, but is not the newest content generation.
    bucket.data["base/linux-cpu/index.json"] = json.dumps(
        [{"path": path} for path in [alias, paths[2], paths[0]]]
    ).encode()
    report = plan(bucket, "bucket", policy)
    assert report["baseVariants"]["linux-cpu"]["previous"] == [paths[0]]
    assert report["baseVariants"]["linux-cpu"]["pending"] == [paths[2]]
    assert not any(row["variant"] for row in report["delete"])
    channel = json.loads(bucket.data["channels/main.json"])
    manifest = json.loads(bucket.data[paths[2] + "/manifest.json"])
    channel["base"]["linux-cpu"]["id"] = manifest["id"]
    channel["base"]["linux-cpu"]["locations"][0]["path"] = paths[2] + "/manifest.json"
    bucket.data["channels/main.json"] = json.dumps(channel).encode()
    report = plan(bucket, "bucket", policy)
    assert report["baseVariants"]["linux-cpu"]["previous"] == [alias]
    assert report["baseVariants"]["linux-cpu"]["pending"] == []
    assert [row["path"] for row in report["delete"] if row["variant"]] == [paths[0]]


def test_base_content_pin_retains_old_payload_without_changing_alias_cleanup(
    base_retention_store, monkeypatch
):
    bucket, policy, paths, alias, releases, tags = base_retention_store
    policy["basePredecessors"] = 0
    identity = json.loads(bucket.data[paths[0] + "/manifest.json"])["id"]
    alias_identity = json.loads(bucket.data[alias + "/manifest.json"])["id"]
    consumers = {
        "sources": {"bases": "current"},
        "pins": [
            {
                "consumer": "Dinkster",
                "repository": "Kosinkadink/Dinkster-Environments",
                "variant": "linux-cpu",
                "identity": value,
            }
            for value in (identity, alias_identity)
        ],
    }
    monkeypatch.setattr(retention, "consumer_pins", lambda: consumers)
    report = plan(bucket, "bucket", policy)
    assert paths[0] in report["keep"]
    assert alias in report["keep"] and paths[1] not in report["keep"]
    before = {
        key: value
        for key, value in bucket.data.items()
        if key.startswith(paths[0] + "/")
    }
    apply(bucket, "bucket", policy, "base", report)
    assert all(bucket.data[key] == value for key, value in before.items())
    tag = "base-linux-cpu-" + paths[0].rsplit("/", 1)[1]
    assert tag in releases and tag in tags
    assert not any(key.startswith(paths[1] + "/") for key in bucket.data)


def test_unresolved_consumer_pin_reports_its_identity_and_refuses_all_deletion(
    retention_store, monkeypatch, tmp_path
):
    bucket, policy, _, github = retention_store
    monkeypatch.setattr(
        retention,
        "consumer_pins",
        lambda: {
            "sources": {},
            "pins": [
                {
                    "consumer": "Desktop",
                    "repository": "Kosinkadink/Dinkster",
                    "tag": "main-missing",
                    "status": "unresolved-deleted",
                }
            ],
        },
    )
    before = deepcopy(bucket.data)
    report = plan(bucket, "bucket", policy)
    assert report["unresolvedConsumerPins"] == [
        {
            "consumer": "Desktop",
            "repository": "Kosinkadink/Dinkster",
            "tag": "main-missing",
            "status": "unresolved-deleted",
        }
    ]
    assert report["delete"] == [] and report["reclaimedBytes"] == 0
    with pytest.raises(ValueError, match="Unresolved consumer pins"):
        apply(bucket, "bucket", policy, "engine", report)
    assert bucket.data == before
    assert github["release"] is not None and github["tag"] is not None
    config = tmp_path / "retention.json"
    config.write_text(json.dumps(policy))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["retention.py", "--config", str(config), "--output", str(output)]
    )
    monkeypatch.setattr("publication.object_store", lambda: bucket)
    monkeypatch.setenv("R2_BUCKET", "bucket")
    retention.main()
    assert (
        json.loads(output.read_bytes())["unresolvedConsumerPins"]
        == report["unresolvedConsumerPins"]
    )
    assert all(bucket.data[key] == value for key, value in before.items())
    assert github["release"] is not None and github["tag"] is not None


@pytest.mark.parametrize("enabled", [False, None, "true", 1])
def test_engine_deletion_needs_explicit_boolean_enablement(
    retention_store, monkeypatch, tmp_path, enabled
):
    bucket, policy, paths, github = retention_store
    policy["engineDeletionEnabled"] = enabled
    if enabled is None:
        del policy["engineDeletionEnabled"]
    report = plan(bucket, "bucket", policy)
    before = deepcopy(bucket.data)
    assert report["engineDeletionEnabled"] is False
    assert [entry["path"] for entry in report["delete"]] == [paths[0]]
    with pytest.raises(ValueError, match="Engine deletion is disabled"):
        apply(bucket, "bucket", policy, "engine", report)
    assert bucket.data == before
    config = tmp_path / "retention.json"
    config.write_text(json.dumps(policy))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retention.py",
            "--scope",
            "engine",
            "--config",
            str(config),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr("publication.object_store", lambda: bucket)
    monkeypatch.setenv("R2_BUCKET", "bucket")
    retention.main()
    assert json.loads(output.read_bytes())["engineDeletionEnabled"] is False
    assert all(bucket.data[key] == value for key, value in before.items())
    assert github["release"] is not None and github["tag"] is not None


def test_publication_lock_serializes_writers_and_releases_on_error(monkeypatch):
    bucket = MemoryBucket()
    key = "retention/publication-lock.json"
    now = [0.0]

    def advance(seconds):
        now[0] += seconds

    monkeypatch.setattr(retention.time, "sleep", advance)
    monkeypatch.setattr(retention.time, "monotonic", lambda: now[0])
    with pytest.raises(RuntimeError, match="publisher failed"):
        with publication_lock(bucket, "bucket"):
            with pytest.raises(RuntimeError, match="Publication lock is held"):
                with publication_lock(bucket, "bucket"):
                    pytest.fail("Second publisher acquired an occupied lock")
            assert json.loads(bucket.data[key])["owner"] is not None
            raise RuntimeError("publisher failed")
    assert json.loads(bucket.data[key]) == {"owner": None}
    with publication_lock(bucket, "bucket"):
        assert json.loads(bucket.data[key])["owner"] is not None
    assert json.loads(bucket.data[key]) == {"owner": None}


@pytest.mark.parametrize("release_at", [650, 14_390, None])
def test_publication_waits_for_a_healthy_long_writer_without_stealing(
    monkeypatch, release_at
):
    bucket = MemoryBucket()
    key = "retention/publication-lock.json"
    original = b'{"owner":"first-publisher"}'
    bucket.data[key] = original
    now = [0.0]

    def advance(seconds):
        assert bucket.data[key] == original
        now[0] += seconds
        if release_at is not None and now[0] >= release_at:
            bucket.data[key] = b'{"owner":null}'

    monkeypatch.setattr(retention.time, "sleep", advance)
    monkeypatch.setattr(retention.time, "monotonic", lambda: now[0])
    if release_at is None:
        with pytest.raises(RuntimeError, match="Publication lock is held"):
            with publication_lock(bucket, "bucket"):
                pytest.fail("A held publication lock must never be stolen")
        assert now[0] == 14_400
        assert bucket.data[key] == original
        assert bucket.writes == 0
    else:
        with publication_lock(bucket, "bucket"):
            assert json.loads(bucket.data[key])["owner"] not in (
                None,
                "first-publisher",
            )
            assert now[0] >= release_at
        assert json.loads(bucket.data[key]) == {"owner": None}


@pytest.mark.parametrize("contended_until", [650, None])
def test_publication_lock_bounds_conditional_write_contention(
    monkeypatch, contended_until
):
    bucket = MemoryBucket()
    now = [0.0]
    put = bucket.put_object

    def advance(seconds):
        assert 0 < seconds <= 10
        now[0] += seconds

    def contend(**kwargs):
        if contended_until is None or now[0] < contended_until:
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": 412}}, "PutObject"
            )
        return put(**kwargs)

    monkeypatch.setattr(retention.time, "sleep", advance)
    monkeypatch.setattr(retention.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(bucket, "put_object", contend)
    if contended_until is None:
        with pytest.raises(RuntimeError, match="Could not acquire publication lock"):
            with publication_lock(bucket, "bucket"):
                pytest.fail("Conditional write contention must not bypass the lock")
        assert now[0] == 14_400
        assert bucket.writes == 0
    else:
        with publication_lock(bucket, "bucket"):
            assert now[0] == 650
        assert json.loads(bucket.data["retention/publication-lock.json"]) == {
            "owner": None
        }
