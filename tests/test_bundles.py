from __future__ import annotations

import copy
import hashlib
import json
import os
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
        package("opencv-contrib-python-headless", "5.0.0.93", pypi, platforms),
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
    assert manifest["platform"] == {
        "os": "linux" if variant == "linux-cpu" else "windows",
        "arch": "x86_64",
    }


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
            "platform",
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
        "R2_ACCESS_KEY_ID": "test-id",
        "R2_SECRET_ACCESS_KEY": "test-secret",
        "R2_ACCOUNT_ID": "test-account",
        "R2_BUCKET": "test",
        "R2_PUBLIC_BASE_URL": "https://example.invalid",
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


@pytest.mark.parametrize("failure", ["branch", "hash", "extras", "unlisted"])
def test_publication_rejects_wrong_source_or_damaged_assets(
    config: dict,
    lock: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    monkeypatch.setenv("ENABLE_BASE_PUBLICATION", "true")
    monkeypatch.setenv("GITHUB_REPOSITORY", "Kosinkadink/Dinkster-Environments")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    manifest = plan(config, "linux-cpu", lock, 11)
    payload = b"archive volume"
    manifest["parts"] = [
        {
            "filename": "base.7z.001",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    (tmp_path / "base.7z.001").write_bytes(payload)
    extras = {
        "format": "dinkster.extras/1",
        "base": manifest["id"],
        "variant": manifest["variant"],
        "wheels": [],
    }
    if failure == "branch":
        monkeypatch.setenv("GITHUB_REF", "refs/heads/base-bundles-552")
        error = "restricted to main"
    elif failure == "hash":
        (tmp_path / "base.7z.001").write_bytes(b"changed volume")
        error = "size/hash mismatch"
    elif failure == "extras":
        extras["base"] = "sha256:" + "f" * 64
        error = "Extras index does not match"
    else:
        (tmp_path / "unexpected.txt").write_text("not a release asset")
        error = "unlisted assets"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "extras.json").write_text(json.dumps(extras))
    with pytest.raises(ValueError, match=error):
        publish(tmp_path)


def test_index_provenance_does_not_claim_missing_lock_coverage(
    config: dict, lock: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    resolved = copy.deepcopy(lock)
    resolved["package"] = [
        p
        for p in resolved["package"]
        if p["source"]["registry"].endswith("/cpu")
        or p["name"] not in ("torch", "torchvision")
    ]
    lock["package"] = [p for p in lock["package"] if p["name"] == "numpy"]
    manifest = plan(config, "win-cpu", lock, 11, resolved)
    provenance = {
        w["name"]: (w["verification"], w["lockedVersion"]) for w in manifest["wheels"]
    }
    assert provenance == {
        "numpy": ("lock", "2.5.1"),
        "torch": ("index", None),
        "torchvision": ("index", None),
        "opencv-contrib-python-headless": ("index", None),
    }
    assert "torch-2.14.1+cpu" in capsys.readouterr().out
    torch = next(p for p in resolved["package"] if p["name"] == "torch")
    pypi_torch = package("torch", "2.14.1", "https://pypi.org/simple", ["win_amd64"])
    lock["package"].append(pypi_torch)
    manifest = plan(config, "win-cpu", lock, 11, resolved)
    selected = next(w for w in manifest["wheels"] if w["name"] == "torch")
    assert selected["verification"] == "index"
    assert selected["lockedVersion"] == "2.14.1"
    assert selected["version"] == torch["version"] == "2.14.1+cpu"


def test_index_cannot_override_locked_hash_or_version(config: dict, lock: dict) -> None:
    resolved = copy.deepcopy(lock)
    resolved["package"] = [
        p
        for p in resolved["package"]
        if p["source"]["registry"].endswith("/cpu")
        or p["name"] not in ("torch", "torchvision")
    ]
    resolved["package"][0]["wheels"][0]["hash"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="index artifact differs from Dinkster lock"):
        plan(config, "win-cpu", lock, 11, resolved)
    resolved["package"][0] = package(
        "numpy", "2.5.2", "https://pypi.org/simple", ["win_amd64"]
    )
    with pytest.raises(ValueError, match="does not match base pin"):
        plan(config, "win-cpu", lock, 11, resolved)


@pytest.mark.parametrize("changed", [False, True])
def test_cli_skip_reads_torch_versions_without_resolving(
    config: dict, lock: dict, tmp_path: Path, changed: bool
) -> None:
    manifest = plan(config, "linux-cpu", lock, 11)
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps(manifest))
    for package in lock["package"]:
        if package["name"] in ("torch", "torchvision"):
            for artifact in package["wheels"]:
                artifact["hash"] = "sha256:" + "e" * 64
            if changed and package["name"] == "torch":
                package["version"] = "2.15.0"
    source = tmp_path / "source"
    boundary = source / "packages/dinkster-workers/src/dinkster_workers/boundary.py"
    boundary.parent.mkdir(parents=True)
    boundary.write_text("PROTOCOL_VERSION = 11\n")
    lines = []
    for package in lock["package"]:
        registry = json.dumps(package["source"]["registry"])
        lines.extend(
            [
                "[[package]]",
                f"name = {json.dumps(package['name'])}",
                f"version = {json.dumps(package['version'])}",
                f"source = {{ registry = {registry} }}",
            ]
        )
        for artifact in package["wheels"]:
            lines.append("[[package.wheels]]")
            lines.extend(
                f"{key} = {json.dumps(value)}" for key, value in artifact.items()
            )
    (source / "uv.lock").write_text("\n".join(lines))
    output = tmp_path / "plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bundles.py"),
            "--config",
            str(ROOT / "variants.toml"),
            "plan",
            "--source",
            str(source),
            "--variant",
            "linux-cpu",
            "--previous",
            str(previous),
            "--output",
            str(output),
        ],
        env={**os.environ, "PATH": ""},
        capture_output=True,
        text=True,
    )
    if changed:
        assert result.returncode != 0
        assert "Build skipped" not in result.stdout
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "compare versions and variant index, not lock hashes" in result.stdout
        assert json.loads(output.read_text()) == manifest
