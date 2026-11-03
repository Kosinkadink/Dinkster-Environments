from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest
import zstandard
from botocore.exceptions import ClientError

from bundles import (
    base_unchanged,
    build,
    extract,
    fetch,
    manifest_id,
    plan,
    verify,
)
from publication import (
    mirror,
    mirror_enabled,
    previous,
    publish,
    relabel_channels,
    write_index,
)

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
    platforms = ["win_amd64", "manylinux_2_28_x86_64", "macosx_14_0_arm64"]
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
    packages.extend(
        [
            package("torch", "2.14.1", pypi, platforms),
            package("torchvision", "0.29.1", pypi, platforms),
        ]
    )
    return {"package": packages}


@pytest.mark.parametrize(
    ("variant", "cuda", "platform", "nvidia"),
    [
        ("linux-cpu", "cpu", "manylinux_2_28_x86_64", None),
        (
            "linux-nvidia",
            "cu132",
            "manylinux_2_28_x86_64",
            {
                "minimumDriver": "580",
                "minimumComputeCapability": "7.5",
                "maximumExclusiveComputeCapability": None,
            },
        ),
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
    assert manifest["archive"] == {"format": "tar+zstd", "parts": []}
    assert manifest["workerProtocol"] == 11
    if platform == "win_amd64":
        assert (
            manifest["interpreter"]["sqlite"]
            == config["interpreters"]["windows"]["sqlite"]
        )
    else:
        assert "sqlite" not in manifest["interpreter"]
    assert manifest["platform"] == {
        "os": "linux" if platform == "manylinux_2_28_x86_64" else "windows",
        "arch": "x86_64",
    }


@pytest.mark.parametrize(
    ("variant", "suffix", "torch_version", "torch_index"),
    [
        ("linux-cpu", "", "2.14.1+cpu", "https://download.pytorch.org/whl/cpu"),
        (
            "win-nvidia-cu126",
            "-cu126",
            "2.14.1+cu126",
            "https://download.pytorch.org/whl/cu126",
        ),
        ("mac-arm64", "", "2.14.1", "https://pypi.org/simple"),
    ],
)
def test_plan_records_readable_build_metadata_outside_identity(
    config: dict,
    lock: dict,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    suffix: str,
    torch_version: str,
    torch_index: str,
) -> None:
    class BuildTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 30, 23, 59, 58, tzinfo=UTC)

    monkeypatch.setattr("bundles.datetime", BuildTime)
    config["source"] = {"repository": "example/engine", "commit": "a" * 40}
    manifest = plan(config, variant, lock, 11)
    assert manifest["builtAt"] == "2026-10-30T23:59:58Z"
    assert manifest["torchVersion"] == torch_version
    assert manifest["torchIndex"] == torch_index
    assert manifest["lockSource"] == {
        "repository": "https://github.com/example/engine",
        "commit": "a" * 40,
    }
    assert manifest["label"] == (
        "2026-10-30-py3.14.10-torch2.14.1" + suffix + "-" + manifest["id"][7:15]
    )
    for field in ("label", "builtAt", "torchVersion", "torchIndex", "lockSource"):
        manifest[field] = "different metadata"
        assert manifest_id(manifest) == manifest["id"]
        del manifest[field]
    assert base_unchanged(config, variant, lock, 11, manifest)


def test_linux_nvidia_manifest_keeps_complete_cuda_closure(config: dict) -> None:
    resolved = json.loads(
        (ROOT / "tests/fixtures/linux-nvidia-resolved.json").read_text()
    )
    manifest = plan(config, "linux-nvidia", resolved, 12, resolved)
    names = {wheel["name"] for wheel in manifest["wheels"]}
    assert names == {
        "cuda-bindings",
        "cuda-pathfinder",
        "cuda-toolkit",
        "filelock",
        "fsspec",
        "jinja2",
        "markupsafe",
        "mpmath",
        "networkx",
        "numpy",
        "nvidia-cublas",
        "nvidia-cuda-cupti",
        "nvidia-cuda-nvrtc",
        "nvidia-cuda-runtime",
        "nvidia-cudnn-cu13",
        "nvidia-cufft",
        "nvidia-cufile",
        "nvidia-curand",
        "nvidia-cusolver",
        "nvidia-cusparse",
        "nvidia-cusparselt-cu13",
        "nvidia-nccl-cu13",
        "nvidia-nvjitlink",
        "nvidia-nvshmem-cu13",
        "nvidia-nvtx",
        "opencv-contrib-python-headless",
        "pillow",
        "setuptools",
        "sympy",
        "torch",
        "torchvision",
        "triton",
        "typing-extensions",
    }
    assert set(manifest["replaces"]) == names
    assert manifest["python"] == config["base"]["python"]
    assert (
        manifest["interpreter"]["sha256"] == config["interpreters"]["linux"]["sha256"]
    )
    assert "sqlite" not in manifest["interpreter"]
    assert next(w for w in manifest["wheels"] if w["name"] == "torch")["version"] == (
        "2.14.1+cu132"
    )
    assert base_unchanged(config, "linux-nvidia", resolved, 12, manifest)
    nvtx = next(p for p in resolved["package"] if p["name"] == "nvidia-nvtx")
    nvtx["wheels"][0]["url"] = (
        "https://example.org/nvidia_nvtx-13.2.86-py3-none-manylinux_2_29_x86_64.whl"
    )
    with pytest.raises(ValueError, match="no compatible locked wheel for nvidia-nvtx"):
        plan(config, "linux-nvidia", resolved, 12, resolved)


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
            "replaces",
            "workerProtocol",
            "nvidia",
        )
    }
    payload["archive"] = {"format": "tar+zstd"}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    assert original["id"] == "sha256:" + expected
    original["archive"]["parts"] = [
        {"filename": "base.tar.zst.001", "sha256": "b" * 64, "size": 99}
    ]
    assert manifest_id(original) == original["id"]
    retired = dict(original, archive={"format": "7z", "parts": []})
    assert manifest_id(retired) != original["id"]
    assert not base_unchanged(config, "linux-cpu", lock, 11, retired)
    del retired["archive"]
    assert not base_unchanged(config, "linux-cpu", lock, 11, retired)
    with pytest.raises(ValueError, match="Unsupported archive format"):
        build(retired, Path("unused"))


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


def test_macos_arm64_selects_locked_pypi_wheels_and_skips_unchanged(
    config: dict, lock: dict
) -> None:
    numpy = lock["package"][0]
    mac_numpy = copy.deepcopy(numpy)
    numpy["resolution-markers"] = ["sys_platform != 'darwin'"]
    mac_numpy["resolution-markers"] = [
        "sys_platform == 'darwin' and platform_machine == 'arm64' "
        "and platform_system == 'Darwin' and os_name == 'posix'"
    ]
    mac_numpy["wheels"][2]["hash"] = "sha256:" + "c" * 64
    lock["package"].append(mac_numpy)
    manifest = plan(config, "mac-arm64", lock, 11)
    assert manifest["platform"] == {"os": "macos", "arch": "arm64"}
    assert manifest["interpreter"]["filename"] == (
        "cpython-3.14.10+20261026-aarch64-apple-darwin-install_only_stripped.tar.gz"
    )
    assert manifest["interpreter"]["executable"] == "python/bin/python3"
    assert manifest["nvidia"] is None
    assert len(manifest["wheels"]) == 4
    assert all(
        wheel["filename"].endswith("-macosx_14_0_arm64.whl")
        and wheel["index"] == "https://pypi.org/simple"
        and wheel["verification"] == "lock"
        for wheel in manifest["wheels"]
    )
    assert (
        next(w for w in manifest["wheels"] if w["name"] == "torch")["version"]
        == "2.14.1"
    )
    assert (
        next(w for w in manifest["wheels"] if w["name"] == "numpy")["sha256"]
        == "c" * 64
    )
    assert base_unchanged(config, "mac-arm64", lock, 11, manifest)
    wrong_arch = copy.deepcopy(manifest)
    wrong_arch["platform"]["arch"] = "x86_64"
    wrong_arch["id"] = manifest_id(wrong_arch)
    assert not base_unchanged(config, "mac-arm64", lock, 11, wrong_arch)


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


def test_previous_selects_newest_labelled_release_for_exact_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "base-win-nvidia-" + "a" * 64
    newest = "base-win-nvidia-2026-10-31-py3.14.10-torch2.14.1-cu132-bbbbbbbb"
    listing = [
        {"tag_name": old, "published_at": "2026-10-29", "draft": False},
        {"tag_name": newest, "published_at": "2026-10-31", "draft": False},
        {"tag_name": newest + ".", "published_at": "2026-11-01", "draft": False},
        {
            "tag_name": newest.replace("win-nvidia-", "win-nvidia-cu126-"),
            "published_at": "2026-11-02",
            "draft": False,
        },
        {"tag_name": old, "published_at": "2026-11-03", "draft": True},
    ]
    monkeypatch.setattr("publication.releases", lambda: listing)

    def download(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert args[3] == newest
        Path(args[args.index("--output") + 1]).write_bytes(b"selected manifest")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("publication.subprocess.run", download)
    output = tmp_path / "previous.json"
    previous("win-nvidia", output)
    assert output.read_bytes() == b"selected manifest"


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
    config: dict, lock: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = plan(config, "linux-cpu", lock, 11)
    python = tmp_path / "python"
    (python / "bin").mkdir(parents=True)
    interpreter_script = (
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n'.encode()
    )
    (python / "bin/python3").write_bytes(interpreter_script)
    (python / "bin/python3").chmod(0o755)
    (python / "bin/python").symlink_to("python3")
    (python / "share/A").mkdir(parents=True)
    (python / "share/a").mkdir()
    (python / "share/A/upper").write_bytes(b"upper")
    (python / "share/a/lower").write_bytes(b"lower")
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
    monkeypatch.setattr("bundles.PART_SIZE", 401)
    first = build(manifest, tmp_path / "first")
    retained = json.loads((tmp_path / "first/manifest.json").read_text())
    second = build(retained, tmp_path / "second")
    assert second["id"] == first["id"] == manifest["id"]
    for field in ("label", "builtAt", "torchVersion", "torchIndex", "lockSource"):
        assert second[field] == first[field] == manifest[field]
    extras = json.loads((tmp_path / "second/extras.json").read_text())
    assert extras == {
        "format": "dinkster.extras/1",
        "base": manifest["id"],
        "variant": "linux-cpu",
        "wheels": [],
    }
    assert len(second["archive"]["parts"]) > 1
    for part in second["archive"]["parts"]:
        path = tmp_path / "second" / part["filename"]
        assert part["size"] == path.stat().st_size <= 401
        assert part["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    extracted = tmp_path / "extracted"
    extract(tmp_path / "second", second, extracted)
    assert (extracted / "python/bin").is_dir()
    assert (extracted / "python/bin/python3").read_bytes() == interpreter_script
    assert os.access(extracted / "python/bin/python3", os.X_OK)
    assert (extracted / "python/bin/python").is_symlink()
    assert (extracted / "python/bin/python").readlink() == Path("python3")
    assert (extracted / "python/share/A/upper").read_bytes() == b"upper"
    assert (extracted / "python/share/a/lower").read_bytes() == b"lower"
    for index, wheel in enumerate(manifest["wheels"]):
        assert (extracted / "wheels" / wheel["filename"]).read_bytes() == (
            f"fixture wheel {index}".encode()
        )
    verify(tmp_path / "second")
    with (tmp_path / "second/base.tar.zst.001").open("ab") as stream:
        stream.write(b"changed archive")
    with pytest.raises(ValueError, match="Archive part size/hash mismatch"):
        verify(tmp_path / "second")


@pytest.mark.parametrize(
    ("name", "kind", "target"),
    [
        ("python/../../escape", tarfile.REGTYPE, ""),
        ("/python/absolute", tarfile.REGTYPE, ""),
        ("python/link", tarfile.SYMTYPE, "../../escape"),
        ("python/link", tarfile.SYMTYPE, "/absolute"),
        ("python/hard", tarfile.LNKTYPE, "python/bin/python3"),
        ("python/device", tarfile.CHRTYPE, ""),
    ],
)
def test_tar_contract_refuses_unsafe_members(
    tmp_path: Path, name: str, kind: bytes, target: str
) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = target
        archive.addfile(member)
    data = zstandard.ZstdCompressor().compress(raw.getvalue())
    (tmp_path / "base.tar.zst.001").write_bytes(data)
    manifest = {
        "archive": {
            "format": "tar+zstd",
            "parts": [
                {
                    "filename": "base.tar.zst.001",
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            ],
        }
    }
    with pytest.raises((ValueError, tarfile.FilterError)):
        extract(tmp_path, manifest, tmp_path / "extracted")
    assert not (tmp_path / "escape").exists()


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


def test_publication_rejects_other_repository_before_any_external_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/other-repository")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    with pytest.raises(ValueError, match="restricted to the Dinkster-Environments"):
        publish(tmp_path)


@pytest.mark.parametrize("github_archives_present", [False, True])
def test_publication_does_not_prune_archives_before_channel_aware_retention(
    config: dict,
    lock: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    github_archives_present: bool,
) -> None:
    manifest = plan(config, "win-cpu", lock, 11)
    manifest.pop("label")
    payload = b"archive"
    manifest["archive"]["parts"] = [
        {
            "filename": "base.tar.zst.001",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    (tmp_path / "base.tar.zst.001").write_bytes(payload)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "extras.json").write_text(
        json.dumps(
            {
                "format": "dinkster.extras/1",
                "base": manifest["id"],
                "variant": "win-cpu",
                "wheels": [],
            }
        )
    )
    releases = []
    objects = set()
    github_assets = set()
    for number in range(12):
        base_id = manifest["id"][7:] if number == 11 else f"{number:064x}"
        archive_name = (
            "base.7z.001" if number % 2 and number != 11 else "base.tar.zst.001"
        )
        releases.append(
            {
                "tag_name": f"base-win-cpu-{base_id}",
                "published_at": f"2026-10-{number + 1:02d}T00:00:00Z",
                "draft": False,
                "assets": [
                    {
                        "id": number,
                        "name": archive_name,
                    }
                ]
                if github_archives_present
                else [],
            }
        )
        if github_archives_present:
            github_assets.add(number)
        for filename in (
            archive_name,
            "manifest.json",
            "extras.json",
        ):
            objects.add(f"s3://test/base/win-cpu/{base_id}/{filename}")
    unrelated = "s3://test/base/win-nvidia/unrelated/base.7z.001"
    objects.add(unrelated)
    environment = {
        "GITHUB_REPOSITORY": "Kosinkadink/Dinkster-Environments",
        "GITHUB_REF": "refs/heads/main",
        "R2_ACCESS_KEY_ID": "test-id",
        "R2_SECRET_ACCESS_KEY": "test-value",
        "R2_ACCOUNT_ID": "test-account",
        "R2_BUCKET": "test",
        "R2_PUBLIC_BASE_URL": "https://example.invalid",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("publication.releases", lambda: releases)
    monkeypatch.setattr("publication.object_store", lambda: Bucket({}))
    before = objects.copy(), github_assets.copy()

    def transport(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if args[:3] == ["aws", "s3", "rm"]:
            objects.difference_update(
                key
                for key in list(objects)
                if key.startswith(args[3])
                and key.endswith(("base.7z.001", "base.tar.zst.001"))
            )
        elif args[:2] == ["gh", "api"] and "DELETE" in args:
            github_assets.remove(int(args[-1].rsplit("/", 1)[-1]))
        else:
            raise AssertionError("Unexpected publication transport")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("publication.subprocess.run", transport)
    publish(tmp_path)
    assert (objects, github_assets) == before


@pytest.mark.parametrize("failure", ["branch", "hash", "extras", "unlisted"])
def test_publication_rejects_wrong_source_or_damaged_assets(
    config: dict,
    lock: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "Kosinkadink/Dinkster-Environments")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    manifest = plan(config, "linux-cpu", lock, 11)
    payload = b"archive volume"
    manifest["archive"]["parts"] = [
        {
            "filename": "base.tar.zst.001",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    (tmp_path / "base.tar.zst.001").write_bytes(payload)
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
        (tmp_path / "base.tar.zst.001").write_bytes(b"changed volume")
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


class Bucket:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.writes: list[str] = []

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str):
        for key, value in sorted(self.objects.items()):
            if key.startswith(Prefix):
                yield {"Contents": [{"Key": key, "Size": len(value)}]}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        self.head_object(Bucket=Bucket, Key=Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self.objects:
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadObject"
            )
        return {"ContentLength": len(self.objects[Key])}

    def upload_file(self, filename: str, bucket: str, key: str, **kwargs: object):
        assert key not in self.objects or key.endswith("/index.json")
        self.objects[key] = Path(filename).read_bytes()
        self.writes.append(key)


@pytest.mark.parametrize(
    "state", ["empty", "partial", "complete", "stale", "corrupt", "forbidden"]
)
@pytest.mark.parametrize("labelled", [False, True])
def test_current_release_mirror_converges_only_verified_absent_objects(
    config: dict,
    lock: dict,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    labelled: bool,
) -> None:
    manifest = plan(config, "linux-cpu", lock, 12)
    name = manifest["id"][7:]
    if labelled:
        name = f"2026-10-31-py3.14.10-torch2.14.1-{name[:8]}"
        manifest["label"] = name
    payload = b"exact published tar+zstd part"
    manifest["archive"]["parts"] = [
        {
            "filename": "base.tar.zst.001",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    extras = {
        "format": "dinkster.extras/1",
        "base": manifest["id"],
        "variant": "linux-cpu",
        "wheels": [],
    }
    source = {
        "manifest.json": json.dumps(manifest).encode(),
        "extras.json": json.dumps(extras).encode(),
        "base.tar.zst.001": payload,
    }
    release = {
        "tag_name": "base-linux-cpu-" + name,
        "draft": False,
        "published_at": "2026-10-29T16:12:06Z",
        "assets": [
            {
                "name": name,
                "size": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            }
            for name, data in source.items()
        ],
    }
    prefix = f"base/linux-cpu/{name}/"
    objects = {
        prefix + name: data
        for name, data in source.items()
        if state != "empty"
        and (state in ("complete", "stale") or name != "base.tar.zst.001")
    }
    if state == "stale":
        objects["base/linux-cpu/index.json"] = b"stale"
    initial = dict(objects)

    class S3(Bucket):
        def head_object(self, *, Bucket: str, Key: str) -> dict:
            if state == "forbidden" or Key not in objects:
                code = 403 if state == "forbidden" else 404
                raise ClientError(
                    {
                        "Error": {"Code": str(code)},
                        "ResponseMetadata": {"HTTPStatusCode": code},
                    },
                    "HeadObject",
                )
            return {"ContentLength": len(objects[Key])}

    client = S3(objects)
    writes = client.writes
    monkeypatch.setattr("publication.boto3.client", lambda *a, **kw: client)
    monkeypatch.setattr("publication.releases", lambda: [release])

    def github(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert args[:3] == ["gh", "release", "download"]
        name = args[args.index("--pattern") + 1]
        data = source[name]
        if state == "corrupt" and name == "base.tar.zst.001":
            data = b"altered bytes"
        Path(args[args.index("--output") + 1]).write_bytes(data)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("publication.subprocess.run", github)
    for name, value in {
        "GITHUB_REPOSITORY": "Kosinkadink/Dinkster-Environments",
        "GITHUB_REF": "refs/heads/main",
        "R2_ACCESS_KEY_ID": "test-id",
        "R2_SECRET_ACCESS_KEY": "test-secret",
        "R2_ACCOUNT_ID": "test-account",
        "R2_BUCKET": "test-bucket",
        "R2_PUBLIC_BASE_URL": "https://example.invalid",
    }.items():
        monkeypatch.setenv(name, value)
    if state in ("corrupt", "forbidden"):
        with pytest.raises(ValueError if state == "corrupt" else ClientError):
            mirror("linux-cpu")
        assert objects == initial
        assert writes == []
    else:
        mirror("linux-cpu")
        index_key = "base/linux-cpu/index.json"
        assert {key: data for key, data in objects.items() if key != index_key} == {
            prefix + name: data for name, data in source.items()
        }
        rows = json.loads(objects[index_key])
        assert rows[0]["id"] == manifest["id"]
        assert rows[0]["path"] == prefix.rstrip("/")
        assert rows[0]["size"] == sum(map(len, source.values()))
        assert writes == (
            [
                prefix + name
                for name in ("manifest.json", "base.tar.zst.001", "extras.json")
            ]
            if state == "empty"
            else []
            if state in ("complete", "stale")
            else [prefix + "base.tar.zst.001"]
        ) + [index_key]
        converged = dict(objects)
        writes.clear()
        mirror("linux-cpu")
        assert objects == converged
        assert writes == []


@pytest.mark.parametrize(
    "state", ["collision", "orphan", "empty", "same", "unconfigured"]
)
def test_base_publication_checks_identity_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    manifest = json.loads((ROOT / "tests/fixtures/base/linux-cpu.json").read_text())
    manifest["workerProtocol"] = 36246
    manifest["id"] = manifest_id(manifest)
    old = copy.deepcopy(manifest)
    old["workerProtocol"] = 99375
    old["id"] = manifest_id(old)
    assert manifest["id"] != old["id"]
    assert manifest["id"][7:15] == old["id"][7:15] == "00c4343a"
    manifest["label"] = "2026-10-31-py3.14.10-torch2.14.1-00c4343a"
    payload = b"archive"
    manifest["archive"]["parts"] = [
        {
            "filename": "base.tar.zst.001",
            "size": 7,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "base.tar.zst.001").write_bytes(payload)
    (tmp_path / "extras.json").write_text(
        json.dumps(
            {
                "format": "dinkster.extras/1",
                "base": manifest["id"],
                "variant": "linux-cpu",
                "wheels": [],
            }
        )
    )
    published = manifest if state == "same" else old
    prefix = f"base/linux-cpu/{published['id'][7:]}"
    objects = (
        {}
        if state in ("empty", "unconfigured")
        else {
            prefix
            + ("/orphan" if state == "orphan" else "/manifest.json"): b"incomplete"
            if state == "orphan"
            else json.dumps(published).encode()
        }
    )
    client = Bucket(objects)
    monkeypatch.setattr("publication.object_store", lambda: client)
    for key, value in {
        "GITHUB_REPOSITORY": "Kosinkadink/Dinkster-Environments",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": "a" * 40,
        "R2_ACCESS_KEY_ID": "test",
        "R2_SECRET_ACCESS_KEY": "test",
        "R2_ACCOUNT_ID": "test",
        "R2_BUCKET": "test",
        "R2_PUBLIC_BASE_URL": "https://example.invalid",
    }.items():
        monkeypatch.setenv(key, value)

    calls = []

    def transport(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("publication.subprocess.run", transport)
    monkeypatch.setattr("publication.releases", lambda: [])
    if state == "unconfigured":
        monkeypatch.delenv("R2_BUCKET")
    if state in ("empty", "same"):
        publish(tmp_path)
        tag = "base-linux-cpu-2026-10-31-py3.14.10-torch2.14.1-00c4343a"
        assert [args[:4] for args in calls] == [
            ["gh", "release", action, tag] for action in ("create", "upload", "edit")
        ]
    else:
        with pytest.raises(
            ValueError,
            match="R2 configuration" if state == "unconfigured" else "hash8 collision",
        ):
            publish(tmp_path)
        assert calls == []
    assert client.writes == []


@pytest.mark.parametrize(
    "state",
    [
        "new",
        "partial",
        "interrupted",
        "race",
        "corrupt",
        "collision",
        "partial-source",
        "partial-torch",
        "partial-index",
        "partial-label",
        "partial-date",
    ],
)
def test_channel_base_republication_preserves_payload_and_other_channel_fields(
    config: dict, lock: dict, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    config["source"] = {"repository": "example/engine", "commit": "a" * 40}
    manifest = plan(config, "linux-cpu", lock, 11)
    for field in ("label", "builtAt", "torchVersion", "torchIndex", "lockSource"):
        manifest.pop(field)
    payload = b"original immutable archive bytes"
    manifest["archive"]["parts"] = [
        {
            "filename": "base.tar.zst.001",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    identity = manifest["id"]
    old = f"base/linux-cpu/{identity[7:]}"
    source = {
        "manifest.json": json.dumps(manifest).encode(),
        "extras.json": json.dumps(
            {
                "format": "dinkster.extras/1",
                "base": identity,
                "variant": "linux-cpu",
                "wheels": [],
            }
        ).encode(),
        "base.tar.zst.001": payload,
    }
    channel = {
        "format": "dinkster.channel/1",
        "signature": None,
        "engine": {"release": "sha256:" + "e" * 64},
        "minimumDesktopVersion": "0.2.0",
        "base": {
            "linux-cpu": {
                "id": identity,
                "locations": [{"type": "mirror", "path": old + "/manifest.json"}],
            },
            "win-cpu": {"id": "sha256:" + "d" * 64, "locations": []},
        },
    }
    objects = {old + "/" + name: data for name, data in source.items()}
    objects["channels/main.json"] = json.dumps(channel).encode()
    initial = dict(objects)
    tag = "base-linux-cpu-" + identity[7:]
    release = {
        "tag_name": tag,
        "draft": False,
        "assets": [
            {
                "name": name,
                "size": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            }
            for name, data in source.items()
        ],
    }
    published = [release]
    uploads = {}
    if state.startswith("partial") or state == "collision":
        labelled = plan(config, "linux-cpu", lock, 11)
        if state == "collision":
            labelled["workerProtocol"] += 1
            labelled["id"] = manifest_id(labelled)
        labelled["builtAt"] = "2026-10-31T12:34:56Z"
        name = f"2026-10-31-py3.14.10-torch2.14.1-{identity[7:15]}"
        labelled["label"] = name
        labelled["archive"]["parts"] = manifest["archive"]["parts"]
        if state == "partial-source":
            labelled["lockSource"]["commit"] = "f" * 40
        elif state == "partial-torch":
            labelled["torchVersion"] = "9.9.9"
        elif state == "partial-index":
            labelled["torchIndex"] = "https://example.invalid/wrong"
        elif state == "partial-label":
            name = labelled["label"] = name.replace("torch2.14.1", "torch9.9.9")
        elif state == "partial-date":
            labelled["builtAt"] = "not a UTC timestamp"
        objects[f"base/linux-cpu/{name}/manifest.json"] = json.dumps(labelled).encode()

    class Store(Bucket):
        race = state == "race"
        interrupted = state == "interrupted"

        def get_object(self, *, Bucket: str, Key: str) -> dict:
            response = super().get_object(Bucket=Bucket, Key=Key)
            return dict(
                response,
                ContentLength=len(objects[Key]),
                ETag=hashlib.sha256(objects[Key]).hexdigest(),
            )

        def upload_file(self, filename: str, bucket: str, key: str, **kwargs):
            objects[key] = Path(filename).read_bytes()
            self.writes.append(key)

        def copy_object(self, *, Bucket: str, Key: str, CopySource: dict):
            objects[Key] = (
                b"corrupt" if state == "corrupt" else objects[CopySource["Key"]]
            )

        def put_object(self, *, Bucket, Key, Body, IfMatch, **kwargs):
            if self.interrupted:
                self.interrupted = False
                raise RuntimeError("Interrupted before channel movement")
            if self.race:
                self.race = False
                newer = json.loads(objects[Key])
                newer["engine"] = {"release": "sha256:" + "f" * 64}
                objects[Key] = json.dumps(newer).encode()
            if IfMatch != hashlib.sha256(objects[Key]).hexdigest():
                raise ClientError(
                    {"ResponseMetadata": {"HTTPStatusCode": 412}}, "PutObject"
                )
            objects[Key] = Body

    client = Store(objects)
    monkeypatch.setattr("publication.object_store", lambda: client)
    monkeypatch.setattr("publication.releases", lambda: published)
    monkeypatch.setattr(
        "publication.subprocess.check_output",
        lambda *a, **k: (
            '[source]\nrepository="example/engine"\ncommit="' + "a" * 40 + '"\n'
        ),
    )
    for name, value in {
        "GITHUB_REPOSITORY": "Kosinkadink/Dinkster-Environments",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": "b" * 40,
        "R2_ACCESS_KEY_ID": "test",
        "R2_SECRET_ACCESS_KEY": "test",
        "R2_ACCOUNT_ID": "test",
        "R2_BUCKET": "test",
        "R2_PUBLIC_BASE_URL": "https://example.invalid",
    }.items():
        monkeypatch.setenv(name, value)

    def github(args, **kwargs):
        assert args[:2] == ["gh", "release"]
        action, requested = args[2:4]
        if action == "download":
            Path(args[args.index("--output") + 1]).write_bytes(
                source[args[args.index("--pattern") + 1]]
            )
        elif action == "create":
            published.append({"tag_name": requested, "draft": True, "assets": []})
        elif action == "upload":
            uploads[requested] = {
                Path(path).name: Path(path).read_bytes()
                for path in args[args.index("--clobber") + 1 :]
            }
            next(r for r in published if r["tag_name"] == requested)["assets"] = [
                {
                    "name": name,
                    "size": len(data),
                    "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                }
                for name, data in uploads[requested].items()
            ]
        elif action == "edit":
            next(r for r in published if r["tag_name"] == requested)["draft"] = False
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("publication.subprocess.run", github)
    if state in ("collision", "corrupt") or state.startswith("partial-"):
        with pytest.raises(ValueError):
            relabel_channels("linux-cpu")
        assert objects["channels/main.json"] == initial["channels/main.json"]
        assert uploads == {}
        if state.startswith("partial-"):
            assert client.writes == []
    else:
        if state == "interrupted":
            with pytest.raises(RuntimeError, match="Interrupted before channel"):
                relabel_channels("linux-cpu")
            assert objects["channels/main.json"] == initial["channels/main.json"]
        relabel_channels("linux-cpu")
        moved = json.loads(objects["channels/main.json"])
        location = moved["base"]["linux-cpu"]["locations"][0]["path"]
        new = location.removesuffix("/manifest.json")
        labelled = json.loads(objects[location])
        assert new != old
        assert labelled["id"] == identity == manifest_id(labelled)
        assert labelled["archive"] == manifest["archive"]
        if state == "partial":
            assert labelled["builtAt"] == "2026-10-31T12:34:56Z"
            assert labelled["label"].startswith("2026-10-31-")
        assert labelled["lockSource"] == {
            "repository": "https://github.com/example/engine",
            "commit": "a" * 40,
        }
        assert objects[new + "/base.tar.zst.001"] == payload
        assert objects[new + "/extras.json"] == source["extras.json"]
        expected = copy.deepcopy(channel)
        expected["base"]["linux-cpu"]["locations"][0]["path"] = location
        if state == "race":
            expected["engine"] = {"release": "sha256:" + "f" * 64}
        assert moved == expected
        rows = json.loads(objects["base/linux-cpu/index.json"])
        assert rows[0]["path"] == new and rows[0]["label"] == labelled["label"]
        assert rows[0]["builtAt"] == labelled["builtAt"]
        assert (
            uploads["base-linux-cpu-" + labelled["label"]]["base.tar.zst.001"]
            == payload
        )
        after = dict(objects)
        relabel_channels("linux-cpu")
        assert objects == after
    assert all(
        objects[key] == data
        for key, data in initial.items()
        if key != "channels/main.json"
    )


@pytest.mark.parametrize("legacy_archive", [False, True])
def test_base_index_rebuilds_paginated_retained_metadata(
    config: dict,
    lock: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_archive: bool,
) -> None:
    monkeypatch.setenv("R2_BUCKET", "test")
    objects = {"base/linux-cpu/index.json": b"stale"}
    ids = []
    for protocol in (11, 12, 13):
        manifest = plan(config, "linux-cpu", lock, protocol)
        manifest["builtAt"] = f"2026-10-{protocol:02d}T00:00:00Z"
        name = manifest["label"] if protocol != 11 else manifest["id"][7:]
        if protocol == 11:
            for field in ("label", "builtAt", "torchVersion", "torchIndex"):
                manifest.pop(field)
            if legacy_archive:
                del manifest["archive"]
                identity = {
                    key: manifest[key]
                    for key in (
                        "format",
                        "variant",
                        "platform",
                        "python",
                        "interpreter",
                        "wheels",
                        "replaces",
                        "workerProtocol",
                        "nvidia",
                    )
                }
                manifest["id"] = (
                    "sha256:"
                    + hashlib.sha256(
                        json.dumps(
                            identity, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                )
                name = manifest["id"][7:]
                manifest["parts"] = [
                    {"filename": "base.7z.001", "size": 137, "sha256": "a" * 64}
                ]
        prefix = f"base/linux-cpu/{name}"
        objects[prefix + "/manifest.json"] = json.dumps(manifest).encode()
        if protocol != 13:
            objects[prefix + "/extras.json"] = json.dumps(
                {
                    "format": "dinkster.extras/1",
                    "base": manifest["id"],
                    "variant": "linux-cpu",
                    "wheels": [],
                }
            ).encode()
            ids.append(manifest["id"])
    output = tmp_path / "index.json"
    write_index(Bucket(objects), "linux-cpu", output)
    rows = json.loads(output.read_bytes())
    assert [row["id"] for row in rows] == ids[::-1]
    assert rows[0]["builtAt"] == "2026-10-12T00:00:00Z"
    assert rows[1]["label"] is None
    assert rows[1]["torchVersion"] == "2.14.1+cpu"
    assert rows[1]["torchIndex"] == "https://download.pytorch.org/whl/cpu"
    legacy_prefix = rows[1]["path"]
    assert rows[1]["size"] == (
        len(objects[legacy_prefix + "/manifest.json"])
        + len(objects[legacy_prefix + "/extras.json"])
        + (137 if legacy_archive else 0)
    )
    original = output.read_bytes()
    write_index(Bucket(objects), "linux-cpu", output)
    assert output.read_bytes() == original
    if legacy_archive:
        legacy_key = legacy_prefix + "/manifest.json"
        damaged = json.loads(objects[legacy_key])
        damaged["python"] = "3.14.9"
        objects[legacy_key] = json.dumps(damaged).encode()
        with pytest.raises(ValueError, match="directory differs"):
            write_index(Bucket(objects), "linux-cpu", output)


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
    lock["package"].append({"name": "engine", "dependencies": [{"name": "numpy"}]})
    resolved["package"][0]["wheels"][0]["hash"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="index artifact differs from Dinkster lock"):
        plan(config, "win-cpu", lock, 11, resolved)
    resolved["package"][0] = package(
        "numpy", "2.5.2", "https://pypi.org/simple", ["win_amd64"]
    )
    with pytest.raises(ValueError, match="does not match base pin"):
        plan(config, "win-cpu", lock, 11, resolved)


@pytest.mark.parametrize(
    ("consumer_marker", "optional", "refused"),
    [
        (None, False, False),
        ("sys_platform == 'linux'", False, True),
        ("sys_platform == 'win32'", False, False),
        ("sys_platform == 'linux'", True, True),
    ],
)
def test_variant_closure_diverges_only_outside_engine_interface(
    config: dict, lock: dict, consumer_marker: str | None, optional: bool, refused: bool
) -> None:
    runtime = package(
        "cuda-toolkit", "13.0.3.0", "https://pypi.org/simple", ["manylinux_2_28_x86_64"]
    )
    lock["package"].append(runtime)
    for entry in lock["package"]:
        if entry["name"] == "torch":
            entry.setdefault("dependencies", []).append({"name": "cuda-toolkit"})
    resolved = copy.deepcopy(lock)
    resolved["package"] = [
        p
        for p in resolved["package"]
        if p["source"]["registry"].endswith("/cu132")
        or p["name"] not in ("torch", "torchvision")
    ]
    resolved["package"][-1] = package(
        "cuda-toolkit", "13.2.2", "https://pypi.org/simple", ["manylinux_2_28_x86_64"]
    )
    if consumer_marker:
        dependency = {"name": "cuda-toolkit", "marker": consumer_marker}
        consumer: dict = {"name": "engine-consumer"}
        if optional:
            consumer["optional-dependencies"] = {"cuda": [dependency]}
        else:
            consumer["dependencies"] = [dependency]
        lock["package"].append(consumer)
    if refused:
        with pytest.raises(ValueError, match="cuda-toolkit: interface version differs"):
            plan(config, "linux-nvidia", lock, 11, resolved)
        return
    manifest = plan(config, "linux-nvidia", lock, 11, resolved)
    selected = next(w for w in manifest["wheels"] if w["name"] == "cuda-toolkit")
    assert selected["version"] == "13.2.2"
    assert selected["verification"] == "index"
    assert selected["lockedVersion"] == "13.0.3.0"
    assert base_unchanged(config, "linux-nvidia", lock, 11, manifest)
    runtime["version"] = "13.1.0"
    assert base_unchanged(config, "linux-nvidia", lock, 11, manifest)
    lock["package"].append(
        {"name": "new-engine-consumer", "dependencies": [{"name": "cuda-toolkit"}]}
    )
    assert not base_unchanged(config, "linux-nvidia", lock, 11, manifest)


@pytest.mark.parametrize("variant", ["linux-cpu", "linux-nvidia"])
@pytest.mark.parametrize(
    "changed",
    [None, "version", "membership", "engine-only", "python", "base-artifact", "legacy"],
)
def test_cli_skip_reads_torch_versions_without_resolving(
    config: dict, lock: dict, tmp_path: Path, changed: str | None, variant: str
) -> None:
    lock["package"].append(
        {
            "name": "engine",
            "version": "1.0",
            "source": {"virtual": "."},
            "dependencies": [{"name": "numpy"}, {"name": "torch"}],
        }
    )
    manifest = plan(config, variant, lock, 11)
    if changed == "legacy":
        for field in ("label", "builtAt", "torchVersion", "torchIndex", "lockSource"):
            del manifest[field]
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps(manifest))
    config_path = tmp_path / "variants.toml"
    config_text = (ROOT / "variants.toml").read_text()
    if changed == "python":
        config_text = config_text.replace('python = "3.14.10"', 'python = "3.14.11"')
    config_path.write_text(config_text)
    if changed == "engine-only":
        for name in ("comfy-kitchen", "comfy-aimdo", "engine-only"):
            lock["package"].append(
                {
                    "name": name,
                    "version": "99.0",
                    "source": {"registry": "https://pypi.org"},
                }
            )
    for package in lock["package"]:
        if changed == "base-artifact" and package["name"] == "numpy":
            for artifact in package["wheels"]:
                artifact["hash"] = "sha256:" + "f" * 64
        if package["name"] in ("torch", "torchvision"):
            for artifact in package["wheels"]:
                artifact["hash"] = "sha256:" + "e" * 64
            if changed == "version" and package["name"] == "torch":
                package["version"] = "2.15.0"
    source = tmp_path / "source"
    boundary = source / "packages/dinkster-workers/src/dinkster_workers/boundary.py"
    boundary.parent.mkdir(parents=True)
    boundary.write_text("PROTOCOL_VERSION = 11\n")
    lines = []
    for package in lock["package"]:
        source_text = ", ".join(
            f"{key} = {json.dumps(value)}" for key, value in package["source"].items()
        )
        lines.extend(
            [
                "[[package]]",
                f"name = {json.dumps(package['name'])}",
                f"version = {json.dumps(package['version'])}",
                f"source = {{ {source_text} }}",
            ]
        )
        if changed == "membership" and package["name"] == "torch":
            lines.append('dependencies = [{name = "new-runtime"}]')
        elif package.get("dependencies"):
            dependencies = ", ".join(
                f"{{name = {json.dumps(dependency['name'])}}}"
                for dependency in package["dependencies"]
            )
            lines.append(f"dependencies = [{dependencies}]")
        for artifact in package.get("wheels", []):
            lines.append("[[package.wheels]]")
            lines.extend(
                f"{key} = {json.dumps(value)}" for key, value in artifact.items()
            )
    (source / "uv.lock").write_text("\n".join(lines))
    output = tmp_path / "plan.json"
    workflow_output = tmp_path / "workflow-output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bundles.py"),
            "--config",
            str(config_path),
            "plan",
            "--source",
            str(source),
            "--variant",
            variant,
            "--previous",
            str(previous),
            "--output",
            str(output),
        ],
        env={**os.environ, "PATH": "", "GITHUB_OUTPUT": str(workflow_output)},
        capture_output=True,
        text=True,
    )
    if changed in ("version", "membership", "python", "base-artifact"):
        assert result.returncode != 0
        assert "Build skipped" not in result.stdout
        assert not output.exists()
        assert not workflow_output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "compare versions and variant index, not lock hashes" in result.stdout
        assert json.loads(output.read_text()) == manifest
        assert workflow_output.read_text().splitlines() == [
            "skip=true",
            f"id={manifest['id'][7:]}",
        ]


def test_cpu_manifest_replaces_universal_cuda_closure_and_requested_extras(
    config: dict, lock: dict
) -> None:
    resolved = copy.deepcopy(lock)
    resolved["package"] = [
        p
        for p in resolved["package"]
        if p["source"]["registry"].endswith("/cpu")
        or p["name"] not in ("torch", "torchvision")
    ]
    torch = package("torch", "2.14.1", "https://pypi.org/simple", [])
    torch["dependencies"] = [
        {
            "name": "cuda-toolkit",
            "extra": ["cublas"],
            "marker": "sys_platform == 'linux'",
        }
    ]
    toolkit = package("cuda-toolkit", "13.0", "https://pypi.org/simple", [])
    toolkit["optional-dependencies"] = {
        "cublas": [{"name": "vendor_runtime"}],
        "unrequested": [{"name": "unrequested-runtime"}],
    }
    runtime = package("vendor-runtime", "13.0", "https://pypi.org/simple", [])
    runtime["dependencies"] = [{"name": "cuda-toolkit", "extra": ["cublas"]}]
    lock["package"] = [p for p in lock["package"] if p["name"] != "torch"]
    lock["package"].extend([torch, toolkit, runtime])
    original = plan(config, "win-cpu", lock, 11, resolved)
    assert original["replaces"] == [
        "cuda-toolkit",
        "numpy",
        "opencv-contrib-python-headless",
        "torch",
        "torchvision",
        "vendor-runtime",
    ]
    assert {w["name"] for w in original["wheels"]} == {
        "numpy",
        "opencv-contrib-python-headless",
        "torch",
        "torchvision",
    }
    lock["package"].append(package("engine-only", "1.0", "https://pypi.org/simple", []))
    assert plan(config, "win-cpu", lock, 11, resolved)["id"] == original["id"]
    toolkit["optional-dependencies"]["cublas"].append({"name": "new-runtime"})
    updated = plan(config, "win-cpu", lock, 11, resolved)
    assert "new-runtime" in updated["replaces"]
    assert updated["wheels"] == original["wheels"]
    assert updated["id"] != original["id"]
