"""GitHub-first base publication, optional R2 mirroring, and archive retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

from bundles import ARCHIVE_FORMAT, base_release_name, manifest_id, retired_bases

REPOSITORY = "Kosinkadink/Dinkster-Environments"


def releases() -> list[dict]:
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", f"repos/{REPOSITORY}/releases"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [release for page in json.loads(result.stdout) for release in page]


def previous(variant: str, output: Path) -> None:
    matches = sorted(
        [
            r
            for r in releases()
            if not r["draft"] and base_release_name(r["tag_name"], variant) is not None
        ],
        key=lambda release: release["published_at"],
        reverse=True,
    )
    if not matches:
        print(f"No published base exists for {variant}")
        return
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            matches[0]["tag_name"],
            "--repo",
            REPOSITORY,
            "--pattern",
            "manifest.json",
            "--output",
            str(output),
        ],
        check=True,
    )


def mirror_enabled(environment: dict[str, str]) -> bool:
    return all(
        environment.get(name)
        for name in (
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_ACCOUNT_ID",
            "R2_BUCKET",
            "R2_PUBLIC_BASE_URL",
        )
    )


def download_asset(release: dict, asset: dict, directory: Path) -> Path:
    path = directory / asset["name"]
    if path.name != asset["name"]:
        raise ValueError("Release asset must be a basename")
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            release["tag_name"],
            "--repo",
            REPOSITORY,
            "--pattern",
            asset["name"],
            "--output",
            str(path),
        ],
        check=True,
    )
    with path.open("rb") as stream:
        digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != asset["size"] or digest != asset["digest"]:
        raise ValueError("GitHub release asset size/hash mismatch")
    return path


def mirror(variant: str) -> None:
    if (
        os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
    ):
        raise ValueError("Mirror convergence is restricted to repository main")
    if not mirror_enabled(dict(os.environ)):
        print("R2 mirror skipped: credentials or configuration are absent")
        return
    release = max(
        (
            r
            for r in releases()
            if not r["draft"] and base_release_name(r["tag_name"], variant) is not None
        ),
        key=lambda r: r["published_at"],
    )
    assets = {asset["name"]: asset for asset in release["assets"]}
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 4},
            connect_timeout=120,
            read_timeout=120,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=4,
        preferred_transfer_client="classic",
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        metadata = {
            name: download_asset(release, assets[name], directory)
            for name in ("manifest.json", "extras.json")
        }
        manifest = json.loads(metadata["manifest.json"].read_text())
        name = base_release_name(release["tag_name"], variant)
        if (
            name is None
            or manifest_id(manifest) != manifest["id"]
            or manifest["variant"] != variant
            or name not in (manifest["id"][7:], manifest.get("label"))
            or (name != manifest["id"][7:] and not name.endswith(manifest["id"][7:15]))
            or manifest["archive"]["format"] != ARCHIVE_FORMAT
            or not manifest["archive"]["parts"]
        ):
            raise ValueError("Release manifest identity mismatch")
        extras = json.loads(metadata["extras.json"].read_text())
        if (
            extras["format"] != "dinkster.extras/1"
            or extras["base"] != manifest["id"]
            or extras["variant"] != variant
        ):
            raise ValueError("Extras index does not match base")
        expected = [assets[name] for name in metadata]
        for number, part in enumerate(manifest["archive"]["parts"], 1):
            if part["filename"] != f"base.tar.zst.{number:03d}":
                raise ValueError("Archive part names must be sequential")
            asset = assets[part["filename"]]
            if (
                asset["size"] != part["size"]
                or not 0 < part["size"] < 2_000_000_000
                or asset["digest"] != "sha256:" + part["sha256"]
            ):
                raise ValueError("Release part differs from manifest size/hash")
            expected.append(asset)
        for asset in expected:
            asset_name = asset["name"]
            key = f"base/{variant}/{name}/{asset_name}"
            try:
                client.head_object(Bucket=os.environ["R2_BUCKET"], Key=key)
            except ClientError as error:
                if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                    raise
            else:
                print(f"R2 object already present: {key}")
                continue
            path = metadata.get(asset_name) or download_asset(release, asset, directory)
            client.upload_file(str(path), os.environ["R2_BUCKET"], key, Config=transfer)
            print(f"R2 missing object uploaded: {key}")
    print(f"R2 mirror converged: {variant} {manifest['id']}")


def publish(directory: Path) -> None:
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise ValueError(
            "Publication is restricted to the Dinkster-Environments repository"
        )
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise ValueError("Public base publication is restricted to main")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("archive", {}).get("format") != ARCHIVE_FORMAT:
        raise ValueError("Unsupported archive format")
    if manifest_id(manifest) != manifest["id"]:
        raise ValueError("Manifest id mismatch")
    if not manifest["archive"]["parts"]:
        raise ValueError("Publication requires built archive parts")
    assets = [directory / "manifest.json", directory / "extras.json"]
    extras = json.loads(assets[1].read_text())
    if (
        extras["format"] != "dinkster.extras/1"
        or extras["base"] != manifest["id"]
        or extras["variant"] != manifest["variant"]
    ):
        raise ValueError("Extras index does not match base")
    for number, part in enumerate(manifest["archive"]["parts"], 1):
        if part["filename"] != f"base.tar.zst.{number:03d}":
            raise ValueError("Archive part names must be sequential")
        path = directory / part["filename"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            path.stat().st_size != part["size"]
            or not 0 < part["size"] < 2_000_000_000
            or digest != part["sha256"]
        ):
            raise ValueError("Archive part size/hash mismatch")
        assets.append(path)
    if set(directory.iterdir()) != set(assets):
        raise ValueError("Publication directory contains unlisted assets")
    tag = f"base-{manifest['variant']}-{manifest['id'][7:]}"
    existing = next((r for r in releases() if r["tag_name"] == tag), None)
    if existing is None:
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                REPOSITORY,
                "--draft",
                "--target",
                os.environ["GITHUB_SHA"],
                "--title",
                tag,
                "--notes",
                "Hash-verified Dinkster base interpreter and variant wheels.",
            ],
            check=True,
        )
    if existing is None or existing["draft"]:
        subprocess.run(
            [
                "gh",
                "release",
                "upload",
                tag,
                "--repo",
                REPOSITORY,
                "--clobber",
                *[str(path) for path in assets],
            ],
            check=True,
        )
        subprocess.run(
            [
                "gh",
                "release",
                "edit",
                tag,
                "--repo",
                REPOSITORY,
                "--draft=false",
                "--latest=false",
            ],
            check=True,
        )
    mirror = mirror_enabled(dict(os.environ))
    endpoint = f"https://{os.environ.get('R2_ACCOUNT_ID', '')}.r2.cloudflarestorage.com"
    aws_environment = dict(os.environ)
    if mirror:
        aws_environment.update(
            AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
            AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"],
        )
    published = releases()
    for release in retired_bases(published, manifest["variant"]):
        if mirror:
            name = base_release_name(release["tag_name"], manifest["variant"])
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "rm",
                    f"s3://{os.environ['R2_BUCKET']}/base/{manifest['variant']}/{name}/",
                    "--recursive",
                    "--exclude",
                    "*",
                    "--include",
                    "base.7z.*",
                    "--include",
                    "base.tar.zst.*",
                    "--endpoint-url",
                    endpoint,
                ],
                env=aws_environment,
                check=True,
            )
        for asset in release["assets"]:
            if asset["name"].startswith(("base.7z.", "base.tar.zst.")):
                subprocess.run(
                    [
                        "gh",
                        "api",
                        "--method",
                        "DELETE",
                        f"repos/{REPOSITORY}/releases/assets/{asset['id']}",
                    ],
                    check=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("previous")
    read.add_argument("--variant", required=True)
    read.add_argument("--output", type=Path, required=True)
    upload = commands.add_parser("publish")
    upload.add_argument("--directory", type=Path, required=True)
    converge = commands.add_parser("mirror")
    converge.add_argument("--variant", required=True)
    args = parser.parse_args()
    if args.command == "previous":
        previous(args.variant, args.output)
    elif args.command == "publish":
        publish(args.directory)
    else:
        mirror(args.variant)


if __name__ == "__main__":
    main()
