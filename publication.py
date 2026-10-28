"""GitHub-first base publication, optional R2 mirroring, and archive retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from bundles import manifest_id, retired_bases

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
    prefix = f"base-{variant}-"
    matches = sorted(
        [
            r
            for r in releases()
            if not r["draft"]
            and r["tag_name"].startswith(prefix)
            and len(r["tag_name"].removeprefix(prefix)) == 64
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


def publish(directory: Path) -> None:
    if os.environ.get("ENABLE_BASE_PUBLICATION") != "true":
        raise ValueError(
            "Public base publication requires explicit repository approval"
        )
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise ValueError(
            "Publication is restricted to the Dinkster-Environments repository"
        )
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise ValueError("Public base publication is restricted to main")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest_id(manifest) != manifest["id"]:
        raise ValueError("Manifest id mismatch")
    if not manifest["parts"]:
        raise ValueError("Publication requires built archive parts")
    assets = [directory / "manifest.json", directory / "extras.json"]
    extras = json.loads(assets[1].read_text())
    if (
        extras["format"] != "dinkster.extras/1"
        or extras["base"] != manifest["id"]
        or extras["variant"] != manifest["variant"]
    ):
        raise ValueError("Extras index does not match base")
    for number, part in enumerate(manifest["parts"], 1):
        if part["filename"] != f"base.7z.{number:03d}":
            raise ValueError("Archive part names must be sequential")
        path = directory / part["filename"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            path.stat().st_size != part["size"]
            or part["size"] >= 2_000_000_000
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
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(directory),
                f"s3://{os.environ['R2_BUCKET']}/base/{manifest['variant']}/{manifest['id'][7:]}/",
                "--recursive",
                "--endpoint-url",
                endpoint,
            ],
            env=aws_environment,
            check=True,
        )
        print(
            f"R2 mirror published: {os.environ['R2_PUBLIC_BASE_URL'].rstrip('/')}/"
            f"base/{manifest['variant']}/{manifest['id'][7:]}/"
        )
    else:
        print(
            "R2 mirror skipped: credentials, account, bucket or public URL are absent"
        )
    published = releases()
    for release in retired_bases(published, manifest["variant"]):
        if mirror:
            base_id = release["tag_name"].rsplit("-", 1)[1]
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "rm",
                    f"s3://{os.environ['R2_BUCKET']}/base/{manifest['variant']}/{base_id}/",
                    "--recursive",
                    "--exclude",
                    "*",
                    "--include",
                    "base.7z.*",
                    "--endpoint-url",
                    endpoint,
                ],
                env=aws_environment,
                check=True,
            )
        for asset in release["assets"]:
            if asset["name"].startswith("base.7z."):
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
    args = parser.parse_args()
    if args.command == "previous":
        previous(args.variant, args.output)
    else:
        publish(args.directory)


if __name__ == "__main__":
    main()
