# Development and reconstruction

Use uv 0.11.29. No external archiver is needed. The Dinkster source checkout
must be at the intended commit; no engine imports or engine dependencies are
needed. From this repository, with that checkout at `../Dinkster`:

```sh
uv sync --locked
uv run --locked python bundles.py plan --source ../Dinkster --variant linux-cpu --output plan.json
uv run --locked python bundles.py build --manifest plan.json --output dist/linux-cpu
uv run --locked python bundles.py verify --directory dist/linux-cpu
```

To reconstruct a tar+zstd base, use the saved `manifest.json` instead of
`plan.json` in the build command. Keep the upstream interpreter and wheel
artifacts available, and use an empty output directory. For incremental
planning, pass `--previous <published-manifest>` to `plan`.

Windows builds require native Windows and the `sqlite-build` dependency group:

```sh
uv sync --locked --group sqlite-build
```

Before moving torchvision, check cp314 wheels on every configured torch index,
not just PyPI. The cu126 variant retains torch 2.14.1 when other variants move.
Add a hardware target in `variants.toml` rather than another builder. Optional
compiled pack wheels belong in the per-variant extras index, initially empty;
retain exact torch/CUDA/Python compatibility and the manifest wheel fields.

Drop the Windows SQLite override when the pinned interpreter includes an
upstream release with the mapped-shm recovery fix; retain the functional gate.
Do not replace it with an application retry or database journal-mode change.
See `windows_sqlite.py`, its native workflow gate and the pinned provenance in
`variants.toml`.

## Fixtures

`tests/fixtures/base/<variant>.json` are consumer plans, not downloadable
archives. Build parts before using them for installation tests. Their historical
Dinkster lock is commit
`5b84da719dc5888b9b3d1dfd7213e4841001b5a0` (landed 2026-10-29 05:44 Pacific),
independent of the publisher source pin.

`tests/fixtures/linux-nvidia-resolved.json` records the resolver's complete
33-distribution cu132/Python 3.14.10 selection at the base root pins, without
engine constraints. It is not a downloadable base or evidence of compatibility
with the pinned engine lock.

## Checks

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright
uv run --locked python -m pytest -q
```

For a full suite on the shared X570 workspace, use its admission launcher:

```sh
PATH=/home/kosin/.local/bin:$PATH /home/kosin/comfy-vibe-station/run_counted_suite.sh uv run --locked python -m pytest -q
```

Hosted CI runs pytest directly. Bundle contracts are in `bundles.py` and
`tests/test_bundles.py`, not this operating guide.
