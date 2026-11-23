# Retention and PyPI cleanup

## Review retention before applying it

Use the authenticated entry point for a read-only report:

```sh
uv run --locked python signed_retention.py --scope engine --dry-run --old-naming --output retention-report.json
```

Review its authenticated channel selections, consumer pins, complete inventory,
candidates and byte counts before changing deletion configuration. Register new
release-backed consumers with retention before relying on a publication.
Enable `engineDeletionEnabled` only after comparing the current listing with
every current consumer pin. Tools deletion remains disabled until
[the tools issue](https://github.com/Kosinkadink/comfy-vibe-station/issues/633)
has its own recorded approval and dry-run listing; do not reuse engine/base
approval. Application requires both recorded approval and a posted dry-run
listing in `retention.json`.

The `release-retention` workflow artifact carries the report. Engine pruning
runs in the engine workflow and base pruning here; there is no independent
archive-count or GitHub-only pruner. Git source and Actions artifact pins are
outside this policy. `consumer-pins.json` is the public producer-read reference
for private consumers; registry artifact locks have no engine/base release
dependency. See `retention.py`, `signed_retention.py` and
`tests/test_retention.py` for selection and retry safeguards.

## Exceptional deletion approvals

The six base paths explicitly retired in
[the base retirement ruling](https://github.com/Kosinkadink/comfy-vibe-station/issues/622#issuecomment-6891849825)
are not granted predecessor grace; old tagged engines do not protect those
bases for fresh installs.

The sole approved manifest-less deletion is mirror-only:
`engine/ba724c0e9d6db6e0f5c701040132ce482dbf700e74d73fd441c5b5090a54cee5`.
Its [approval](https://github.com/Kosinkadink/comfy-vibe-station/issues/923#issuecomment-6891744766)
is bound to the authenticated 165-object, 623,381,189-byte inventory in
[the evidence packet](https://github.com/Kosinkadink/comfy-vibe-station/issues/911#issuecomment-6889860050).
Do not generalize this exception to other incomplete directories or use it to
delete a GitHub release/tag or rewrite an engine index. Other manifest-less
publications remain protected. The exact approval and inventory guards are
implemented in `retention.py`.

## PyPI development wheels

First produce the authenticated read-only retention report above. Then run:

```sh
uv run --locked python pypi_inventory.py \
  --retention-report retention-report.json \
  --output retained-pypi-pins.json

uv run --locked python pypi_cleanup.py \
  --inventory retained-pypi-pins.json \
  --output pypi-cleanup-report.json \
  --markdown-output pypi-cleanup-report.md
```

The issue owner posts the Markdown for the user to review and delete the listed
releases through the PyPI web UI. The cleanup command has no deletion or yank
operation and does not use a GitHub token; the inventory producer uses read-only
GitHub access. Registry packs and `dinkster-pack-helpers` are outside this
listing. Selection and completeness checks live in `pypi_inventory.py`,
`pypi_cleanup.py` and their tests.
