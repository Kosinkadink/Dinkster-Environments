from __future__ import annotations

from copy import deepcopy

import pytest

from retention import keep_paths


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
    kept = keep_paths(publications, [channel], engine_count=2, predecessors=1)
    assert kept == {
        "engine/build-0",
        "engine/build-1",
        "engine/build-3",
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
    assert "engine/build-4" in keep_paths(
        publications, [channel], engine_count=0, predecessors=0
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
            "publishedAt": f"2026-11-{6 - number:02d}T00:00:00Z",
        }
        publications.append(alias)
    kept = keep_paths(publications, [channel], engine_count=0, predecessors=1)
    assert f"base/linux-cpu/{3:064x}" in kept
    assert "base/linux-cpu/label-3" in kept
    assert "base/linux-cpu/label-2" in kept
    assert f"base/linux-cpu/{2:064x}" in kept
    channel["base"]["linux-cpu"]["locations"][0]["path"] = (
        "base/linux-cpu/label-3/manifest.json"
    )
    kept = keep_paths(publications, [channel], engine_count=0, predecessors=1)
    assert f"base/linux-cpu/{3:064x}" in kept
    kept = keep_paths(
        publications,
        [channel],
        engine_count=0,
        predecessors=1,
        verified_aliases=frozenset({"base/linux-cpu/label-3"}),
    )
    assert f"base/linux-cpu/{3:064x}" not in kept
    assert "base/linux-cpu/label-2" in kept


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
        keep_paths(
            publications,
            [] if damage == "no-channels" else [channel],
            engine_count=10,
            predecessors=1,
        )
