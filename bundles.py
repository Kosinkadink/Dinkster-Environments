"""Build bases from locked versions and hash-verified variant index wheels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

IDENTITY_FIELDS = (
    "format",
    "variant",
    "platform",
    "python",
    "interpreter",
    "wheels",
    "workerProtocol",
    "nvidia",
)
PART_SIZE = 1_900_000_000


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def manifest_id(manifest: dict) -> str:
    payload = {key: manifest[key] for key in IDENTITY_FIELDS}
    return "sha256:" + hashlib.sha256(canonical(payload)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value) + b"\n")


def worker_protocol(source: Path) -> int:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PROTOCOL_VERSION"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if type(value) is int and value > 0:
                return value
    raise ValueError("Workers source must declare an integer PROTOCOL_VERSION")


def target_environment(python: str, windows: bool) -> dict[str, str]:
    return {
        "implementation_name": "cpython",
        "implementation_version": python,
        "os_name": "nt" if windows else "posix",
        "platform_machine": "AMD64" if windows else "x86_64",
        "platform_release": "",
        "platform_system": "Windows" if windows else "Linux",
        "platform_version": "",
        "python_full_version": python,
        "platform_python_implementation": "CPython",
        "python_version": ".".join(python.split(".")[:2]),
        "sys_platform": "win32" if windows else "linux",
        "extra": "",
    }


def applicable_packages(
    lock: dict, environment: dict[str, str]
) -> dict[str, list[dict]]:
    packages: dict[str, list[dict]] = {}
    for package in lock["package"]:
        markers = package.get("resolution-markers", [])
        if not markers or any(
            Marker(marker).evaluate(environment) for marker in markers
        ):
            packages.setdefault(canonicalize_name(package["name"]), []).append(package)
    return packages


def base_unchanged(
    config: dict, variant: str, lock: dict, protocol: int, previous: dict
) -> bool:
    if manifest_id(previous) != previous["id"]:
        raise ValueError("Previous manifest id does not match its payload")
    base = config["base"]
    target = config["variants"][variant]
    if (
        previous["variant"] != variant
        or previous["platform"] != {"os": target["os"], "arch": "x86_64"}
        or previous["python"] != base["python"]
        or previous["workerProtocol"] != protocol
        or previous["interpreter"]["release"] != base["release"]
        or any(
            previous["interpreter"].get(key) != value
            for key, value in config["interpreters"][target["os"]].items()
        )
        or previous["nvidia"]
        != (
            {
                key: target.get(key)
                for key in (
                    "minimumDriver",
                    "minimumComputeCapability",
                    "maximumExclusiveComputeCapability",
                )
            }
            if target["cuda"]
            else None
        )
    ):
        return False
    locked = applicable_packages(
        lock, target_environment(base["python"], target["os"] == "windows")
    )
    wheels = {wheel["name"]: wheel for wheel in previous["wheels"]}
    if not set(base["packages"]).issubset(wheels):
        return False
    for name, wheel in wheels.items():
        version = Version(wheel["version"]).base_version
        if name in base["packages"] and version != base[name]:
            return False
        entries = locked.get(name, [])
        if not entries and wheel["lockedVersion"] is not None:
            return False
        if entries and {Version(p["version"]).base_version for p in entries} != {
            version
        }:
            return False
        if name in ("torch", "torchvision"):
            if wheel["index"].rstrip("/") != target["torch-index"].rstrip("/"):
                return False
            continue
        if entries and not any(
            urllib.parse.unquote(urllib.parse.urlsplit(artifact["url"]).path).rsplit(
                "/", 1
            )[-1]
            == wheel["filename"]
            and artifact["hash"] == "sha256:" + wheel["sha256"]
            and artifact["size"] == wheel["size"]
            for entry in entries
            for artifact in entry.get("wheels", [])
        ):
            return False
    return True


def resolve_variant(config: dict, variant: str, lock: dict) -> dict:
    """Resolve only on the publisher; clients use the finished manifest offline."""
    base = config["base"]
    target = config["variants"][variant]
    environment = target_environment(base["python"], target["os"] == "windows")
    locked = applicable_packages(lock, environment)
    constraints = []
    for name, packages in locked.items():
        versions = {p["version"] for p in packages if "registry" in p["source"]}
        if len(versions) > 1:
            raise ValueError(f"Ambiguous target lock versions for {name}: {versions}")
        if versions:
            constraints.append(f"{name}=={versions.pop()}")
    requirements = "\n".join(f"{name}=={base[name]}" for name in base["packages"])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "requirements.in").write_text(requirements)
        (root / "constraints.txt").write_text("\n".join(constraints))
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                str(root / "requirements.in"),
                "--constraints",
                str(root / "constraints.txt"),
                "--format",
                "pylock.toml",
                "--output-file",
                str(root / "pylock.toml"),
                "--python-version",
                base["python"],
                "--python-platform",
                "x86_64-pc-windows-msvc"
                if target["os"] == "windows"
                else "x86_64-manylinux_2_28",
                "--torch-backend",
                target["cuda"] or "cpu",
                "--only-binary",
                ":all:",
                "--no-sources",
                "--no-config",
                "--quiet",
            ],
            check=True,
        )
        resolved = tomllib.loads((root / "pylock.toml").read_text())
    packages = []
    for package in resolved["packages"]:
        wheels = []
        for wheel in package["wheels"]:
            size = wheel.get("size")
            if size is None:
                request = urllib.request.Request(wheel["url"], method="HEAD")
                with urllib.request.urlopen(request, timeout=120) as response:
                    size = int(response.headers["Content-Length"])
            wheels.append(
                {
                    "url": wheel["url"],
                    "size": size,
                    "hash": "sha256:" + wheel["hashes"]["sha256"],
                }
            )
        packages.append(
            {
                "name": package["name"],
                "version": package["version"],
                "source": {
                    "registry": target["torch-index"]
                    if package["name"] in ("torch", "torchvision")
                    else "https://pypi.org/simple"
                },
                "wheels": wheels,
            }
        )
    return {"package": packages}


def plan(
    config: dict, variant: str, lock: dict, protocol: int, resolved: dict | None = None
) -> dict:
    base = config["base"]
    target = config["variants"][variant]
    windows = target["os"] == "windows"
    python = base["python"]
    major, minor, *_ = (int(part) for part in python.split("."))
    platforms = (
        ["win_amd64"]
        if windows
        else [f"manylinux_2_{n}_x86_64" for n in range(28, 16, -1)]
        + ["manylinux2014_x86_64", "linux_x86_64"]
    )
    tags = list(cpython_tags((major, minor), platforms=platforms)) + list(
        compatible_tags(
            (major, minor), interpreter=f"cp{major}{minor}", platforms=platforms
        )
    )
    priority = {tag: i for i, tag in enumerate(tags)}
    environment = target_environment(python, windows)

    def applies(marker: str) -> bool:
        return Marker(marker).evaluate(environment)

    packages = applicable_packages(
        resolved if resolved is not None else lock, environment
    )
    locked = applicable_packages(lock, environment)

    selected: dict[str, dict] = {}
    pending = [{"name": name} for name in (packages if resolved else base["packages"])]
    while pending:
        dependency = pending.pop()
        if "marker" in dependency and not applies(dependency["marker"]):
            continue
        name = canonicalize_name(dependency["name"])
        candidates = packages.get(name, [])
        if "version" in dependency:
            candidates = [
                p for p in candidates if p["version"] == dependency["version"]
            ]
        if "source" in dependency:
            candidates = [p for p in candidates if p["source"] == dependency["source"]]
        if name in ("torch", "torchvision"):
            candidates = [
                p
                for p in candidates
                if p["source"].get("registry", "").rstrip("/")
                == target["torch-index"].rstrip("/")
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"{variant}: expected one locked {name}, found {len(candidates)}; "
                "update the Dinkster lock, never resolve during a base build"
            )
        package = candidates[0]
        if name in selected:
            if package != selected[name]:
                raise ValueError(f"Conflicting locked versions for {name}")
            continue
        if name in base and isinstance(base[name], str):
            version = Version(package["version"])
            pin = base[name]
            if version.base_version != pin:
                raise ValueError(f"{name} {version} does not match base pin {pin}")
        if "registry" not in package["source"]:
            raise ValueError(
                f"Base dependency {name} must have a locked registry source"
            )
        selected[name] = package
        pending.extend(package.get("dependencies", []))

    wheels = []
    for name, package in selected.items():
        matches = []
        for wheel in package.get("wheels", []):
            filename = urllib.parse.unquote(
                urllib.parse.urlsplit(wheel["url"]).path.rsplit("/", 1)[-1]
            )
            wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(filename)
            if wheel_name != name or wheel_version != Version(package["version"]):
                raise ValueError(f"Wheel identity differs from lock entry: {filename}")
            ranks = [priority[tag] for tag in wheel_tags if tag in priority]
            if ranks:
                matches.append((min(ranks), filename, wheel))
        if not matches:
            raise ValueError(f"{variant}: no compatible locked wheel for {name}")
        _, filename, wheel = min(matches)
        if not wheel["hash"].startswith("sha256:"):
            raise ValueError(f"{filename}: lock must carry a sha256 hash")
        lock_entries = locked.get(name, [])
        lock_versions = {Version(p["version"]).base_version for p in lock_entries}
        if lock_versions and lock_versions != {
            Version(package["version"]).base_version
        }:
            raise ValueError(f"{name}: variant version differs from Dinkster lock")
        locked_artifacts = [
            artifact
            for entry in lock_entries
            for artifact in entry.get("wheels", [])
            if urllib.parse.unquote(urllib.parse.urlsplit(artifact["url"]).path).rsplit(
                "/", 1
            )[-1]
            == filename
        ]
        if locked_artifacts and not any(
            artifact["hash"] == wheel["hash"] and artifact["size"] == wheel["size"]
            for artifact in locked_artifacts
        ):
            raise ValueError(f"{filename}: index artifact differs from Dinkster lock")
        if (
            lock_entries
            and not locked_artifacts
            and name not in ("torch", "torchvision")
        ):
            raise ValueError(
                f"{filename}: no target artifact in existing Dinkster lock"
            )
        verification = "lock" if locked_artifacts else "index"
        locked_version = lock_entries[0]["version"] if lock_entries else None
        print(
            f"{variant}: {filename}: {verification}-verified; "
            f"Dinkster locked version: {locked_version or 'absent'}"
        )
        wheels.append(
            {
                "name": name,
                "version": package["version"],
                "filename": filename,
                "size": wheel["size"],
                "sha256": wheel["hash"].removeprefix("sha256:"),
                "index": package["source"]["registry"],
                "url": wheel["url"],
                "verification": verification,
                "lockedVersion": locked_version,
            }
        )
    interpreter = dict(config["interpreters"][target["os"]])
    interpreter["release"] = base["release"]
    interpreter["url"] = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        + base["release"]
        + "/"
        + urllib.parse.quote(interpreter["filename"])
    )
    manifest = {
        "format": "dinkster.base/1",
        "variant": variant,
        "platform": {"os": target["os"], "arch": "x86_64"},
        "python": python,
        "interpreter": interpreter,
        "wheels": sorted(wheels, key=lambda wheel: wheel["filename"]),
        "workerProtocol": protocol,
        "nvidia": (
            {
                "minimumDriver": target["minimumDriver"],
                "minimumComputeCapability": target["minimumComputeCapability"],
                "maximumExclusiveComputeCapability": target.get(
                    "maximumExclusiveComputeCapability"
                ),
            }
            if target["cuda"]
            else None
        ),
        "parts": [],
        "signature": None,
    }
    manifest["id"] = manifest_id(manifest)
    return manifest


def fetch(artifact: dict, destination: Path) -> None:
    if Path(artifact["filename"]).name != artifact["filename"]:
        raise ValueError("Artifact filename must be a basename")
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(artifact["url"], timeout=120) as response:
            with destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    if size > artifact["size"]:
                        raise ValueError(
                            f"Download exceeds locked size: {destination.name}"
                        )
                    output.write(chunk)
        if size != artifact["size"] or digest.hexdigest() != artifact["sha256"]:
            raise ValueError(
                f"Download differs from locked size/hash: {destination.name}"
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def build(manifest: dict, output: Path, seven_zip: str) -> dict:
    if manifest["format"] != "dinkster.base/1":
        raise ValueError("Unsupported base manifest format")
    if manifest["signature"] is not None:
        raise ValueError("Signed manifest verification is not implemented")
    if manifest_id(manifest) != manifest["id"]:
        raise ValueError("Manifest id does not match its payload")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        interpreter = staging / "interpreter.tar.gz"
        fetch(manifest["interpreter"], interpreter)
        payload = staging / "payload"
        payload.mkdir()
        with tarfile.open(interpreter) as archive:
            archive.extractall(payload, filter="data")
        if not (payload / manifest["interpreter"]["executable"]).is_file():
            raise ValueError("Interpreter archive does not contain its executable")
        wheels = payload / "wheels"
        wheels.mkdir()
        for wheel in manifest["wheels"]:
            fetch(wheel, wheels / wheel["filename"])
        subprocess.run(
            [
                seven_zip,
                "a",
                "-t7z",
                f"-v{PART_SIZE}b",
                "-mtc=off",
                "-mta=off",
                "-mtr=off",
                "-mx=1",
                str(output.resolve() / "base.7z"),
                "python",
                "wheels",
            ],
            cwd=payload,
            check=True,
        )
    result = dict(manifest)
    result["parts"] = []
    for part in sorted(output.glob("base.7z.*")):
        if part.stat().st_size >= 2_000_000_000:
            raise ValueError("Archive part exceeds GitHub release size limit")
        with part.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result["parts"].append(
            {
                "filename": part.name,
                "size": part.stat().st_size,
                "sha256": digest,
            }
        )
    if not result["parts"]:
        raise ValueError("Archiver produced no split volumes")
    write_json(output / "manifest.json", result)
    write_json(
        output / "extras.json",
        {
            "format": "dinkster.extras/1",
            "variant": result["variant"],
            "base": result["id"],
            "wheels": [],
        },
    )
    return result


def archive_assets_to_remove(releases: list[dict], variant: str) -> list[dict]:
    prefix = f"base-{variant}-"
    bases = sorted(
        (
            release
            for release in releases
            if not release["draft"]
            and release["tag_name"].startswith(prefix)
            and len(release["tag_name"].removeprefix(prefix)) == 64
        ),
        key=lambda release: release["published_at"],
        reverse=True,
    )
    return [
        asset
        for release in bases[10:]
        for asset in release["assets"]
        if asset["name"].startswith("base.7z.")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("variants.toml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("plan")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--variant", required=True)
    prepare.add_argument("--previous", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    archive = subparsers.add_parser("build")
    archive.add_argument("--manifest", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    archive.add_argument("--seven-zip", default=shutil.which("7zz") or "7z")
    args = parser.parse_args()
    if args.command == "plan":
        config = tomllib.loads(args.config.read_text())
        source = args.source
        lock = tomllib.loads((source / "uv.lock").read_text())
        protocol = worker_protocol(
            source / "packages/dinkster-workers/src/dinkster_workers/boundary.py"
        )
        previous = json.loads(args.previous.read_text()) if args.previous else None
        skip = previous is not None and base_unchanged(
            config, args.variant, lock, protocol, previous
        )
        if previous is not None and skip:
            manifest = previous
            print(
                "Build skipped: base-relevant locked versions and metadata unchanged; "
                "torch/torchvision compare versions and variant index, not lock hashes"
            )
        else:
            manifest = plan(
                config,
                args.variant,
                lock,
                protocol,
                resolve_variant(config, args.variant, lock),
            )
            print(f"Base inputs require build: {manifest['id']}")
        write_json(args.output, manifest)
        if path := os.environ.get("GITHUB_OUTPUT"):
            with open(path, "a") as output:
                output.write(f"skip={str(skip).lower()}\nid={manifest['id'][7:]}\n")
    else:
        build(json.loads(args.manifest.read_text()), args.output, args.seven_zip)


if __name__ == "__main__":
    main()
