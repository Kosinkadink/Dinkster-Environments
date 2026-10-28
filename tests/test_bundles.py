from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

from bundles import archive_assets_to_remove, build, fetch, manifest_id, plan
from publication import mirror_enabled, publish

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config() -> dict:
    return tomllib.loads((ROOT / "variants.toml").read_text())


def package(name: str, version: str, index: str, platforms: list[str]) -> dict:
    return {
        "name": name,
        "version": version,
        "source": {"registry": index},
        "wheels": [
            {
                "url": f"https://example.org/{name.replace('-', '_')}-{version}"
                f"-cp314-cp314-{platform}.whl",
                "size": 17,
                "hash": "sha256:" + hashlib.sha256(platform.encode()).hexdigest(),
            }
            for platform in platforms
        ],
    }


@pytest.fixture
def lock(config: dict) -> dict:
    platforms = ["win_amd64", "manylinux_2_28_x86_64"]
    pypi = "https://pypi.org/simple"
    packages = [
        package("numpy", "2.5.1", pypi, platforms),
        package("opencv-contrib-python-headless", "5.0.0", pypi, platforms),
    ]
    for cuda in ("cpu", "cu126", "cu132"):
        index = f"https://download.pytorch.org/whl/{cuda}"
        torch = package("torch", f"2.14.1+{cuda}", index, platforms)
        torch["dependencies"] = [{"name": "numpy"}]
        packages.extend(
            [torch, package("torchvision", f"0.29.1+{cuda}", index, platforms)]
        )
    return {"package": packages}


@pytest.mark.parametrize(
    ("variant", "cuda", "platform", "nvidia"),
    [
        ("linux-cpu", "cpu", "manylinux_2_28_x86_64", None),
        ("win-cpu", "cpu", "win_amd64", None),
        (
            "win-nvidia",
            "cu132",
            "win_amd64",
            {
                "minimumDriver": "580",
                "minimumComputeCapability": "7.5",
                "maximumExclusiveComputeCapability": None,
            },
        ),
        (
            "win-nvidia-cu126",
            "cu126",
            "win_amd64",
            {
                "minimumDriver": "525",
                "minimumComputeCapability": "5.0",
                "maximumExclusiveComputeCapability": "10.0",
            },
        ),
    ],
)
def test_variant_selects_exact_locked_artifacts(
    config: dict,
    lock: dict,
    variant: str,
    cuda: str,
    platform: str,
    nvidia: dict | None,
) -> None:
    manifest = plan(config, variant, lock, 11)
    wheels = manifest["wheels"]
    assert len(wheels) == 4
    assert all(wheel["filename"].endswith(f"-{platform}.whl") for wheel in wheels)
    torch = next(wheel for wheel in wheels if wheel["name"] == "torch")
    assert torch["version"] == f"2.14.1+{cuda}"
    assert torch["index"] == f"https://download.pytorch.org/whl/{cuda}"
    assert torch["sha256"] == hashlib.sha256(platform.encode()).hexdigest()
    assert manifest["nvidia"] == nvidia
    assert manifest["workerProtocol"] == 11


def test_identity_ignores_engine_only_changes_but_tracks_payload(
    config: dict, lock: dict
) -> None:
    original = plan(config, "linux-cpu", lock, 11)
    lock["package"].append(package("engine-only", "1.0", "https://pypi.org", []))
    assert plan(config, "linux-cpu", lock, 11)["id"] == original["id"]
    lock["package"][0]["wheels"][1]["hash"] = "sha256:" + "a" * 64
    assert plan(config, "linux-cpu", lock, 11)["id"] != original["id"]
    assert plan(config, "linux-cpu", lock, 12)["id"] != original["id"]
    payload = {
        key: original[key]
        for key in (
            "format",
            "variant",
            "python",
            "interpreter",
            "wheels",
            "workerProtocol",
            "nvidia",
        )
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    assert original["id"] == "sha256:" + expected
    original["parts"] = [{"filename": "base.7z.001", "sha256": "b" * 64, "size": 99}]
    assert manifest_id(original) == original["id"]


def test_missing_or_wrong_pin_never_resolves(config: dict, lock: dict) -> None:
    lock["package"] = [p for p in lock["package"] if p["name"] != "torch"]
    with pytest.raises(ValueError, match="expected one locked torch, found 0"):
        plan(config, "win-cpu", lock, 11)
    lock["package"].append(
        package(
            "torch", "2.15.0+cpu", "https://download.pytorch.org/whl/cpu", ["win_amd64"]
        )
    )
    with pytest.raises(ValueError, match="does not match base pin"):
        plan(config, "win-cpu", lock, 11)


def test_markers_select_target_not_build_host(config: dict, lock: dict) -> None:
    numpy = lock["package"][0]
    numpy["resolution-markers"] = ["sys_platform == 'win32'"]
    linux_numpy = copy.deepcopy(numpy)
    linux_numpy["resolution-markers"] = ["sys_platform == 'linux'"]
    linux_numpy["wheels"][1]["hash"] = "sha256:" + "b" * 64
    lock["package"].append(linux_numpy)
    win = plan(config, "win-cpu", lock, 11)
    linux = plan(config, "linux-cpu", lock, 11)
    assert next(w for w in win["wheels"] if w["name"] == "numpy")["sha256"] != "b" * 64
    assert (
        next(w for w in linux["wheels"] if w["name"] == "numpy")["sha256"] == "b" * 64
    )


def test_download_rejects_tampering_and_removes_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.whl"
    source.write_bytes(b"verified wheel")
    artifact = {
        "filename": "source.whl",
        "url": source.as_uri(),
        "sha256": hashlib.sha256(b"verified wheel").hexdigest(),
        "size": 14,
    }
    output = tmp_path / "download.whl"
    fetch(artifact, output)
    assert output.read_bytes() == b"verified wheel"
    source.write_bytes(b"modified wheel")
    with pytest.raises(ValueError, match="differs from locked"):
        fetch(artifact, output)
    assert not output.exists()


def test_retention_keeps_ten_newest_and_all_metadata() -> None:
    releases = [
        {
            "tag_name": "base-win-cpu-" + f"{number:064x}",
            "published_at": f"2026-10-{number + 1:02d}T00:00:00Z",
            "draft": False,
            "assets": [
                {"id": number, "name": "base.7z.001"},
                {"id": 100 + number, "name": "manifest.json"},
                {"id": 200 + number, "name": "extras.json"},
            ],
        }
        for number in range(12)
    ]
    assert {a["id"] for a in archive_assets_to_remove(releases, "win-cpu")} == {0, 1}
    assert archive_assets_to_remove(releases, "win-nvidia") == []


def test_build_rejects_bad_manifest_id_before_network(
    config: dict, lock: dict, tmp_path: Path
) -> None:
    manifest = plan(config, "linux-cpu", lock, 11)
    manifest["workerProtocol"] = 100
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bundles.py"),
            "build",
            "--manifest",
            str(path),
            "--output",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Manifest id does not match" in result.stderr
    assert not (tmp_path / "output").exists()


def test_manifest_only_rebuild_preserves_identity_and_archive_payload(
    config: dict, lock: dict, tmp_path: Path
) -> None:
    manifest = plan(config, "linux-cpu", lock, 11)
    python = tmp_path / "python"
    (python / "bin").mkdir(parents=True)
    (python / "bin/python3").write_bytes(b"fixture interpreter")
    interpreter = tmp_path / "interpreter.tar.gz"
    with tarfile.open(interpreter, "w:gz") as archive:
        archive.add(python, arcname="python")
    manifest["interpreter"].update(
        filename=interpreter.name,
        url=interpreter.as_uri(),
        size=interpreter.stat().st_size,
        sha256=hashlib.sha256(interpreter.read_bytes()).hexdigest(),
    )
    for index, wheel in enumerate(manifest["wheels"]):
        path = tmp_path / wheel["filename"]
        payload = f"fixture wheel {index}".encode()
        path.write_bytes(payload)
        wheel.update(
            url=path.as_uri(),
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    manifest["id"] = manifest_id(manifest)
    seven_zip = shutil.which("7zz") or shutil.which("7z")
    assert seven_zip is not None, "Install 7-Zip to run archive contract tests"
    first = build(manifest, tmp_path / "first", seven_zip)
    retained = json.loads((tmp_path / "first/manifest.json").read_text())
    second = build(retained, tmp_path / "second", seven_zip)
    assert second["id"] == first["id"] == manifest["id"]
    extras = json.loads((tmp_path / "second/extras.json").read_text())
    assert extras == {
        "format": "dinkster.extras/1",
        "base": manifest["id"],
        "variant": "linux-cpu",
        "wheels": [],
    }
    for part in second["parts"]:
        path = tmp_path / "second" / part["filename"]
        assert part["size"] == path.stat().st_size < 2_000_000_000
        assert part["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    extracted = tmp_path / "extracted"
    subprocess.run(
        [seven_zip, "x", str(tmp_path / "second/base.7z.001"), f"-o{extracted}"],
        check=True,
    )
    assert (extracted / "python/bin/python3").read_bytes() == b"fixture interpreter"
    for index, wheel in enumerate(manifest["wheels"]):
        assert (extracted / "wheels" / wheel["filename"]).read_bytes() == (
            f"fixture wheel {index}".encode()
        )


def test_mirroring_requires_all_credentials_and_configuration() -> None:
    environment = {
        "AWS_ACCESS_KEY_ID": "test-id",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "R2_ENDPOINT": "https://example.invalid",
        "R2_BUCKET": "test",
    }
    assert mirror_enabled(environment)
    for name in environment:
        assert not mirror_enabled(
            {key: value for key, value in environment.items() if key != name}
        )


def test_publication_disabled_before_any_external_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ENABLE_BASE_PUBLICATION", raising=False)
    with pytest.raises(ValueError, match="requires explicit repository approval"):
        publish(tmp_path)
