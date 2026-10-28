"""GitHub-first base publication, optional R2 mirroring, and archive retention."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from bundles import archive_assets_to_remove, manifest_id

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
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "R2_ENDPOINT",
            "R2_BUCKET",
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
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest_id(manifest) != manifest["id"]:
        raise ValueError("Manifest id mismatch")
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
                "Hash-verified Dinkster base interpreter and locked wheels.",
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
                *[str(path) for path in sorted(directory.iterdir())],
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
    if mirror_enabled(dict(os.environ)):
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(directory),
                f"s3://{os.environ['R2_BUCKET']}/base/{manifest['variant']}/{manifest['id'][7:]}/",
                "--recursive",
                "--endpoint-url",
                os.environ["R2_ENDPOINT"],
            ],
            check=True,
        )
    else:
        print("R2 mirror skipped: credentials, endpoint or bucket are absent")
    published = releases()
    for asset in archive_assets_to_remove(published, manifest["variant"]):
        if mirror_enabled(dict(os.environ)):
            release = next(r for r in published if asset in r["assets"])
            base_id = release["tag_name"].rsplit("-", 1)[1]
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "rm",
                    f"s3://{os.environ['R2_BUCKET']}/base/{manifest['variant']}/{base_id}/{asset['name']}",
                    "--endpoint-url",
                    os.environ["R2_ENDPOINT"],
                ],
                check=True,
            )
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
