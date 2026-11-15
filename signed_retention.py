"""Retain releases using only authenticated Stable and Nightly channels."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
from pathlib import Path

from dinkster_supervisor.environment_channel_signatures import verify_channel

import retention


class _SignedChannels:
    """Expose verified channel payloads to the shared retention algorithms."""

    def __init__(self, client, expected: dict[str, str] | None = None) -> None:
        self.client = client
        self.expected = expected
        self.envelopes: dict[str, str] = {}

    def __getattr__(self, name):
        return getattr(self.client, name)

    def get_object(self, **kwargs):
        response = self.client.get_object(**kwargs)
        key = kwargs["Key"]
        if not key.startswith("channels/"):
            return response
        data = response["Body"].read()
        channel = verify_channel(data, Path(key).stem)
        envelope = data.decode("ascii")
        if self.expected is not None and self.expected.get(key) != envelope:
            raise ValueError("Channels changed since planning; refusing retention")
        if key in self.envelopes and self.envelopes[key] != envelope:
            raise ValueError("Channels changed during inventory; refusing retention")
        self.envelopes[key] = envelope
        return {**response, "Body": io.BytesIO(json.dumps(channel).encode("ascii"))}


def plan(client, bucket: str, policy: dict, *, old_naming: bool = False) -> dict:
    authenticated = _SignedChannels(client)
    report = retention.plan(authenticated, bucket, policy, old_naming=old_naming)
    report["channelEnvelopes"] = authenticated.envelopes
    return report


def apply(client, bucket: str, policy: dict, scope: str, report: dict) -> None:
    authenticated = _SignedChannels(client, report["channelEnvelopes"])
    _, metadata = retention.inventory(authenticated, bucket)
    channels = {
        key: json.loads(data)
        for key, data in metadata.items()
        if key.startswith("channels/")
    }
    if (
        authenticated.envelopes != report["channelEnvelopes"]
        or channels != report["channels"]
    ):
        raise ValueError("Channels changed since planning; refusing retention")
    retention.apply(authenticated, bucket, policy, scope, report)


def main() -> None:
    from publication import object_store

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("retention.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scope", choices=("engine", "base", "tools"), default="base")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--old-naming", action="store_true")
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    policy = json.loads(args.config.read_text(encoding="ascii"))
    client, bucket = object_store(), os.environ["R2_BUCKET"]
    with retention.publication_lock(client, bucket):
        if args.command:
            subprocess.run(args.command, check=True)
            return
        report = plan(client, bucket, policy, old_naming=args.old_naming)
        data = json.dumps(report, indent=2) + "\n"
        print(data, end="", flush=True)
        if args.output:
            args.output.write_text(data, encoding="ascii")
        approval_fields = (
            ("toolsDeletionApproval", "toolsDryRunListing")
            if args.scope == "tools"
            else ("deletionApproval", "dryRunListing")
        )
        if (
            all(policy.get(field) is not None for field in approval_fields)
            and not report["unresolvedConsumerPins"]
            and (args.scope != "engine" or report["engineDeletionEnabled"])
            and (args.scope != "tools" or report["toolsDeletionEnabled"])
            and not args.dry_run
        ):
            apply(client, bucket, policy, args.scope, report)
            after = plan(client, bucket, policy, old_naming=args.old_naming)
            print(
                "Retention after application: "
                + json.dumps(
                    {
                        "bucketBytes": after["bucketBytes"],
                        "bucketObjects": after["bucketObjects"],
                        "channels": after["channels"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
