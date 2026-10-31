from __future__ import annotations

import copy
import fnmatch
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tomllib
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
    retired_bases,
    verify,
)
from publication import mirror, mirror_enabled, publish

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
    assert {r["tag_name"] for r in retired_bases(releases, "win-cpu")} == {
        "base-win-cpu-" + f"{number:064x}" for number in (0, 1)
    }
    assert retired_bases(releases, "win-nvidia") == []


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
def test_publication_retains_ten_mirrored_archives_and_all_manifests(
    config: dict,
    lock: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    github_archives_present: bool,
) -> None:
    manifest = plan(config, "win-cpu", lock, 11)
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

    def transport(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if args[:3] == ["aws", "s3", "cp"]:
            objects.update(args[4] + path.name for path in Path(args[3]).iterdir())
        elif args[:3] == ["aws", "s3", "rm"]:
            prefix = args[3]
            includes = [args[i + 1] for i, arg in enumerate(args) if arg == "--include"]
            exclude = args[args.index("--exclude") + 1] if "--exclude" in args else ""
            for key in list(objects):
                if key == prefix:
                    objects.remove(key)
                elif "--recursive" in args and key.startswith(prefix):
                    relative = key.removeprefix(prefix)
                    if not fnmatch.fnmatch(relative, exclude) or any(
                        fnmatch.fnmatch(relative, include) for include in includes
                    ):
                        objects.remove(key)
        elif args[:2] == ["gh", "api"] and "DELETE" in args:
            github_assets.remove(int(args[-1].rsplit("/", 1)[-1]))
        else:
            raise AssertionError("Unexpected external transport")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("publication.subprocess.run", transport)
    publish(tmp_path)
    assert unrelated in objects
    assert len([key for key in objects if key.endswith("manifest.json")]) == 12
    assert len([key for key in objects if key.endswith("extras.json")]) == 12
    assert (
        len(
            [
                key
                for key in objects
                if key.endswith(("base.7z.001", "base.tar.zst.001"))
            ]
        )
        == 11
    )
    for number in (0, 1):
        for archive_name in ("base.7z.001", "base.tar.zst.001"):
            assert f"s3://test/base/win-cpu/{number:064x}/{archive_name}" not in objects
    assert github_assets == (set(range(2, 12)) if github_archives_present else set())


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


@pytest.mark.parametrize("state", ["partial", "complete", "corrupt", "forbidden"])
def test_current_release_mirror_converges_only_verified_absent_objects(
    config: dict, lock: dict, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    manifest = plan(config, "linux-cpu", lock, 12)
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
        "tag_name": "base-linux-cpu-" + manifest["id"][7:],
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
    prefix = f"base/linux-cpu/{manifest['id'][7:]}/"
    objects = {
        prefix + name: data
        for name, data in source.items()
        if state == "complete" or name != "base.tar.zst.001"
    }
    initial = dict(objects)
    writes = []

    class S3:
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

        def upload_file(
            self, filename: str, bucket: str, key: str, **kwargs: object
        ) -> None:
            assert key not in objects, "Present objects must never be overwritten"
            objects[key] = Path(filename).read_bytes()
            writes.append(key)

    monkeypatch.setattr("publication.boto3.client", lambda *a, **kw: S3())
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
        assert objects == {prefix + name: data for name, data in source.items()}
        assert writes == ([] if state == "complete" else [prefix + "base.tar.zst.001"])


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
    "changed", [None, "version", "membership", "engine-only", "python", "base-artifact"]
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
