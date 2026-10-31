"""Build bases from locked versions and hash-verified variant index wheels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

import zstandard
from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

import windows_sqlite

IDENTITY_FIELDS = (
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
PART_SIZE = 1_900_000_000
ARCHIVE_FORMAT = "tar+zstd"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def manifest_id(manifest: dict) -> str:
    payload = {key: manifest[key] for key in IDENTITY_FIELDS}
    payload["archive"] = {"format": manifest["archive"]["format"]}
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


def target_environment(python: str, target: dict) -> dict[str, str]:
    system = target["os"]
    return {
        "implementation_name": "cpython",
        "implementation_version": python,
        "os_name": "nt" if system == "windows" else "posix",
        "platform_machine": "AMD64"
        if system == "windows"
        else target.get("arch", "x86_64"),
        "platform_release": "",
        "platform_system": {"windows": "Windows", "linux": "Linux", "macos": "Darwin"}[
            system
        ],
        "platform_version": "",
        "python_full_version": python,
        "platform_python_implementation": "CPython",
        "python_version": ".".join(python.split(".")[:2]),
        "sys_platform": {"windows": "win32", "linux": "linux", "macos": "darwin"}[
            system
        ],
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


def replacement_membership(
    lock: dict, roots: list[str], wheels: list[str]
) -> list[str]:
    names = {canonicalize_name(name) for name in wheels}
    pending = [{"name": name} for name in roots]
    visited: set[tuple[int, str]] = set()
    while pending:
        dependency = pending.pop()
        name = canonicalize_name(dependency["name"])
        names.add(name)
        for index, package in enumerate(lock["package"]):
            if canonicalize_name(package["name"]) != name:
                continue
            if "version" in dependency and dependency["version"] != package["version"]:
                continue
            if "source" in dependency and dependency["source"] != package["source"]:
                continue
            for extra in ("", *dependency.get("extra", [])):
                key = (index, extra)
                if key in visited:
                    continue
                visited.add(key)
                pending.extend(
                    package.get("optional-dependencies", {}).get(extra, [])
                    if extra
                    else package.get("dependencies", [])
                )
    return sorted(names)


def interface_names(
    lock: dict, roots: list[str], environment: dict[str, str], wheels: list[str]
) -> set[str]:
    owned = set(replacement_membership(lock, roots, wheels))
    names = set()
    for name, packages in applicable_packages(lock, environment).items():
        if name in owned:
            continue
        for package in packages:
            groups = {"": package.get("dependencies", [])}
            groups.update(package.get("optional-dependencies", {}))
            for extra, dependencies in groups.items():
                for dependency in dependencies:
                    if "marker" in dependency and not Marker(
                        dependency["marker"]
                    ).evaluate(dict(environment, extra=extra)):
                        continue
                    dependency_name = canonicalize_name(dependency["name"])
                    if dependency_name in owned:
                        names.add(dependency_name)
    return names


def base_unchanged(
    config: dict, variant: str, lock: dict, protocol: int, previous: dict
) -> bool:
    if previous.get("archive", {}).get("format") != ARCHIVE_FORMAT:
        return False
    if manifest_id(previous) != previous["id"]:
        raise ValueError("Previous manifest id does not match its payload")
    base = config["base"]
    target = config["variants"][variant]
    if (
        previous["variant"] != variant
        or previous["platform"]
        != {"os": target["os"], "arch": target.get("arch", "x86_64")}
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
    environment = target_environment(base["python"], target)
    locked = applicable_packages(lock, environment)
    wheels = {wheel["name"]: wheel for wheel in previous["wheels"]}
    interfaces = interface_names(lock, base["packages"], environment, list(wheels))
    if not set(base["packages"]).issubset(wheels):
        return False
    if previous["replaces"] != replacement_membership(
        lock, base["packages"], list(wheels)
    ):
        return False
    for name, wheel in wheels.items():
        version = Version(wheel["version"]).base_version
        if name in base["packages"] and version != base[name]:
            return False
        if name in ("torch", "torchvision") and (
            wheel["index"].rstrip("/") != target["torch-index"].rstrip("/")
        ):
            return False
        if name not in interfaces:
            continue
        entries = locked.get(name, [])
        if not entries and wheel["lockedVersion"] is not None:
            return False
        if entries and {Version(p["version"]).public for p in entries} != {
            Version(wheel["version"]).public
        }:
            return False
        if name in ("torch", "torchvision"):
            continue
        artifacts = [
            artifact
            for entry in entries
            for artifact in entry.get("wheels", [])
            if urllib.parse.unquote(urllib.parse.urlsplit(artifact["url"]).path).rsplit(
                "/", 1
            )[-1]
            == wheel["filename"]
        ]
        if artifacts and not any(
            artifact["hash"] == "sha256:" + wheel["sha256"]
            and artifact["size"] == wheel["size"]
            for artifact in artifacts
        ):
            return False
    return True


def resolve_variant(config: dict, variant: str, lock: dict) -> dict:
    """Resolve only on the publisher; clients use the finished manifest offline."""
    base = config["base"]
    target = config["variants"][variant]
    environment = target_environment(base["python"], target)
    locked = applicable_packages(lock, environment)
    interfaces = interface_names(lock, base["packages"], environment, [])
    constraints = []
    for name, packages in locked.items():
        if name not in interfaces:
            continue
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
                {
                    "windows": "x86_64-pc-windows-msvc",
                    "linux": "x86_64-manylinux_2_28",
                    "macos": "aarch64-apple-darwin",
                }[target["os"]],
                *(
                    []
                    if target["torch-index"] == "https://pypi.org/simple"
                    else ["--torch-backend", target["cuda"] or "cpu"]
                ),
                "--only-binary",
                ":all:",
                "--no-sources",
                "--no-config",
                "--quiet",
            ],
            check=True,
            env=dict(os.environ, MACOSX_DEPLOYMENT_TARGET="14.0")
            if target["os"] == "macos"
            else None,
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
    python = base["python"]
    major, minor, *_ = (int(part) for part in python.split("."))
    platforms = {
        "windows": ["win_amd64"],
        "linux": [f"manylinux_2_{n}_x86_64" for n in range(28, 4, -1)]
        + [
            "manylinux2014_x86_64",
            "manylinux2010_x86_64",
            "manylinux1_x86_64",
            "linux_x86_64",
        ],
        "macos": list(mac_platforms((14, 0), "arm64")),
    }[target["os"]]
    tags = list(cpython_tags((major, minor), platforms=platforms)) + list(
        compatible_tags(
            (major, minor), interpreter=f"cp{major}{minor}", platforms=platforms
        )
    )
    priority = {tag: i for i, tag in enumerate(tags)}
    environment = target_environment(python, target)

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

    interfaces = interface_names(lock, base["packages"], environment, list(selected))
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
        lock_versions = {Version(p["version"]).public for p in lock_entries}
        if (
            name in interfaces
            and lock_versions
            and lock_versions != {Version(package["version"]).public}
        ):
            raise ValueError(f"{name}: interface version differs from Dinkster lock")
        locked_artifacts = [
            artifact
            for entry in lock_entries
            for artifact in entry.get("wheels", [])
            if urllib.parse.unquote(urllib.parse.urlsplit(artifact["url"]).path).rsplit(
                "/", 1
            )[-1]
            == filename
        ]
        matches_lock = any(
            artifact["hash"] == wheel["hash"] and artifact["size"] == wheel["size"]
            for artifact in locked_artifacts
        )
        if name in interfaces and locked_artifacts and not matches_lock:
            raise ValueError(f"{filename}: index artifact differs from Dinkster lock")
        verification = "lock" if matches_lock else "index"
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
        "archive": {"format": ARCHIVE_FORMAT, "parts": []},
        "variant": variant,
        "platform": {"os": target["os"], "arch": target.get("arch", "x86_64")},
        "python": python,
        "interpreter": interpreter,
        "wheels": sorted(wheels, key=lambda wheel: wheel["filename"]),
        "replaces": replacement_membership(lock, base["packages"], list(selected)),
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


def safe_member(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo:
    """The data filter also resolves symlink targets against the extracted tree."""
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in member.name
        or not path.parts
        or path.parts[0] not in ("python", "wheels")
        or not (member.isfile() or member.isdir() or member.issym())
    ):
        raise ValueError(f"Unsupported archive member: {member.name}")
    filtered = tarfile.data_filter(member, destination)
    if filtered is None:
        raise ValueError(f"Archive member refused: {member.name}")
    return filtered


def pack(payload: Path, output: Path) -> list[dict]:
    """Stream a portable tar through zstd, then byte-split its single frame."""
    with tempfile.TemporaryFile() as compressed:
        with zstandard.ZstdCompressor(level=3, threads=-1).stream_writer(
            compressed, closefd=False
        ) as writer:
            with tarfile.open(fileobj=writer, mode="w|", dereference=False) as archive:

                def checked(member: tarfile.TarInfo) -> tarfile.TarInfo:
                    safe_member(member, str(payload))
                    return member

                for name in ("python", "wheels"):
                    archive.add(payload / name, arcname=name, filter=checked)
        compressed.seek(0)
        parts = []
        while chunk := compressed.read(min(PART_SIZE, 1024 * 1024)):
            path = output / f"base.tar.zst.{len(parts) + 1:03d}"
            digest = hashlib.sha256()
            size = 0
            with path.open("wb") as stream:
                while chunk:
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if size == PART_SIZE:
                        break
                    chunk = compressed.read(min(PART_SIZE - size, 1024 * 1024))
            parts.append(
                {"filename": path.name, "size": size, "sha256": digest.hexdigest()}
            )
    return parts


def extract(directory: Path, manifest: dict, destination: Path) -> None:
    if manifest.get("archive", {}).get("format") != ARCHIVE_FORMAT:
        raise ValueError("Unsupported archive format")
    if not manifest["archive"]["parts"]:
        raise ValueError("Archive requires parts")
    with tempfile.TemporaryFile() as combined:
        for number, part in enumerate(manifest["archive"]["parts"], 1):
            if part["filename"] != f"base.tar.zst.{number:03d}":
                raise ValueError("Archive part names must be sequential")
            path = directory / part["filename"]
            with path.open("rb") as stream:
                if (
                    path.stat().st_size != part["size"]
                    or not 0 < part["size"] < 2_000_000_000
                    or hashlib.file_digest(stream, "sha256").hexdigest()
                    != part["sha256"]
                ):
                    raise ValueError("Archive part size/hash mismatch")
                stream.seek(0)
                shutil.copyfileobj(stream, combined)
        combined.seek(0)
        with zstandard.ZstdDecompressor().stream_reader(combined) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                archive.extractall(destination, filter=safe_member)


def build(manifest: dict, output: Path) -> dict:
    if manifest["format"] != "dinkster.base/1":
        raise ValueError("Unsupported base manifest format")
    if manifest.get("archive", {}).get("format") != ARCHIVE_FORMAT:
        raise ValueError("Unsupported archive format")
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
        if manifest["platform"]["os"] == "windows":
            if sys.platform != "win32":
                raise ValueError("Windows bases must pass the native mapped-shm gate")
            override = staging / "sqlite"
            metadata = windows_sqlite.build(override)
            if metadata != manifest["interpreter"].get("sqlite"):
                raise ValueError("Built SQLite differs from the manifest identity")
            windows_sqlite.install(
                override, payload / manifest["interpreter"]["executable"]
            )
        wheels = payload / "wheels"
        wheels.mkdir()
        for wheel in manifest["wheels"]:
            fetch(wheel, wheels / wheel["filename"])
        parts = pack(payload, output)
        # Decode our actual output before publication, not only the source tar.
        extract(
            output,
            {"archive": {"format": ARCHIVE_FORMAT, "parts": parts}},
            staging / "checked",
        )
    result = dict(manifest)
    result["archive"] = {"format": ARCHIVE_FORMAT, "parts": parts}
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


def verify(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest_id(manifest) != manifest["id"]:
        raise ValueError("Manifest id does not match its payload")
    host = {"win32": "windows", "linux": "linux", "darwin": "macos"}[sys.platform]
    if manifest["platform"]["os"] != host:
        raise ValueError(
            "Interpreter verification requires the target operating system"
        )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extracted = root / "extracted"
        extract(directory, manifest, extracted)
        for wheel in manifest["wheels"]:
            path = extracted / "wheels" / wheel["filename"]
            with path.open("rb") as stream:
                if (
                    path.stat().st_size != wheel["size"]
                    or hashlib.file_digest(stream, "sha256").hexdigest()
                    != wheel["sha256"]
                ):
                    raise ValueError(f"Extracted wheel size/hash mismatch: {path.name}")
        executable = extracted / manifest["interpreter"]["executable"]
        if not os.access(executable, os.X_OK):
            raise ValueError("Extracted interpreter is not executable")
        subprocess.run(
            [str(executable), "-I", "-c", "import sys, sysconfig"],
            check=True,
            timeout=60,
        )
        if host == "windows":
            metadata = manifest["interpreter"]["sqlite"]
            dll = extracted / "python/DLLs/sqlite3.dll"
            if (
                dll.stat().st_size != metadata["size"]
                or hashlib.sha256(dll.read_bytes()).hexdigest() != metadata["sha256"]
            ):
                raise ValueError("Extracted SQLite differs from the manifest identity")
            subprocess.run(
                [
                    str(executable),
                    "-I",
                    str(Path(windows_sqlite.__file__).resolve()),
                    "probe",
                ],
                check=True,
                timeout=60,
            )
    print(f"tar+zstd verified wheels and executable interpreter: {manifest['id']}")


def retired_bases(releases: list[dict], variant: str) -> list[dict]:
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
    return bases[10:]


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
    check = subparsers.add_parser("verify")
    check.add_argument("--directory", type=Path, required=True)
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
    elif args.command == "build":
        build(json.loads(args.manifest.read_text()), args.output)
    else:
        verify(args.directory)


if __name__ == "__main__":
    main()
