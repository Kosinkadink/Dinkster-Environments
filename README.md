# Dinkster-Environments

Content-addressed Python base bundles for Dinkster hardware variants. Engine
releases and environment installation belong to Dinkster, not this repository.
Base builds never resolve dependencies: every wheel must exist in the named
Dinkster commit's `uv.lock`, with a sha256 and size.

## Variants

`variants.toml` pins Python 3.14 and python-build-standalone, torch 2.14.1,
numpy 2.5 and opencv-contrib-python-headless 5.0. Torchvision is the matching
version resolved together with torch in the Dinkster lock. The dependency closure
of these packages belongs to the base; all other wheels belong to the engine.

| Variant | Platform | Torch index | NVIDIA bounds |
| --- | --- | --- | --- |
| win-nvidia | Windows x64 | cu132 | driver >=580, capability >=7.5 |
| win-nvidia-cu126 | Windows x64 | cu126 | driver >=525, capability >=5.0 and <10.0 |
| win-cpu | Windows x64 | cpu | none |
| linux-cpu | Linux x64, glibc >=2.28 | cpu | none |

The cu126 variant retains torch 2.14.1 when the other variants move. A new
hardware target is another definition, not another builder. Optional compiled
pack wheels belong to the per-variant extras index, initially empty.

## Manifest contract

`manifest.json` uses `format: dinkster.base/1`. All object fields below are
required. Hashes are bare lowercase sha256 hex, except `id` which has a
`sha256:` prefix. Sizes are byte counts.

| Field | Meaning |
| --- | --- |
| id | Content identity as defined below |
| variant | Variant name from variants.toml |
| python | Full interpreter version |
| interpreter | Object with release, filename, url, size, sha256 and executable |
| wheels | Filename-sorted array of name, version, filename, size, sha256, index, url |
| workerProtocol | Positive integer from the named engine commit's workers source |
| nvidia | null for CPU, otherwise minimumDriver, minimumComputeCapability, maximumExclusiveComputeCapability |
| parts | Ordered array of filename, size, sha256 for split archive volumes |
| signature | null; reserved for ed25519 signing |

NVIDIA bounds are decimal strings; the maximum is null when unbounded.
`interpreter.release` is the python-build-standalone release. `url` is the exact
artifact URL, not an index search. Each wheel's `index` is its source registry
from the lock. Keeping both makes manifest-only reconstruction independent of
an index's current resolution. A removed upstream artifact fails verification;
the builder never substitutes another wheel.

Identity is sha256 of the UTF-8 JSON object containing exactly `format`,
`variant`, `python`, `interpreter`, `wheels`, `workerProtocol` and `nvidia`,
with sorted object keys, compact separators, ASCII escaping, and no trailing
newline. `id`, `parts` and `signature` are excluded. Archive metadata cannot
create a circular content address; rebuilding may yield different archive part
hashes while retaining the same verified payload identity. Engine-only lock
changes do not affect identity. Interpreter, target wheel, protocol or hardware
metadata changes do.

The split 7z archive contains `python/` and `wheels/`. The executable is
`python/python.exe` on Windows or `python/bin/python3` on Linux, relative to
the extracted root. Volumes are `base.7z.001` onwards, each strictly below
2,000,000,000 bytes. Clients verify every part before extraction and every
wheel before installing. `manifest.json` is distributed separately.

Adjacent `extras.json` is an object with `format: dinkster.extras/1`, `base`
(the base id), `variant`, and `wheels: []`. Adding compiled wheels must retain
the manifest wheel record fields and exact torch/CUDA/Python compatibility.

GitHub release tags are `base-<variant>-<id hex>`. Mirror paths are
`base/<variant>/<id hex>/`, containing the same named files. Retention removes
only archive volumes from all but the ten newest published bases per variant;
manifests and extras remain indefinitely.

## Local build and reconstruction

Requires uv and 7-Zip (`7z` or `7zz`) on PATH. The Dinkster source checkout must
be at the intended commit; no engine imports or engine dependencies are needed.

```sh
uv sync --locked
uv run --locked python bundles.py plan --source ../Dinkster --variant linux-cpu --output plan.json
uv run --locked python bundles.py build --manifest plan.json --output dist/linux-cpu
```

To reconstruct a retired archive, run the same `build` command with its retained
`manifest.json` instead of `plan.json`. Only that manifest and the upstream
interpreter and wheel artifacts are needed. Its id and every download's size
and sha256 are checked before archiving. The output directory must be empty.
For incremental builds, `plan --previous <published-manifest>` prints an explicit
skip reason and writes `skip=true` to GITHUB_OUTPUT when identity is unchanged.

## Hosted builds and publication

Branch pushes build all four variants on hosted Linux runners; cross-platform
wheel selection never executes the target interpreter. Each new base is rebuilt
from its manifest and uploaded as a workflow artifact. No hosted workflow needs
to be dispatched manually.

Public releases are disabled by default. The repository owner must create the
`base-publication` protected environment with required reviewers and set the
repository variable `ENABLE_BASE_PUBLICATION=true` to approve publication from
main. Agents do not configure those settings or secrets. The job publishes to
GitHub first, then optionally R2. It never publishes from a feature branch.

R2 needs `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` environment secrets, plus
`R2_ENDPOINT` and `R2_BUCKET` variables. The bucket must serve the configured
`releases.dinkster.org` origin. Missing any of these skips R2 with a log message;
configured mirror errors fail the job, rather than pretending the mirror is
healthy. The hosted publication runner needs `gh` and the AWS CLI.

## Development gates

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright
PATH=/home/kosin/.local/bin:$PATH /home/kosin/comfy-vibe-station/run_counted_suite.sh uv run --locked python -m pytest -q
```

The counted launcher is the workspace's local full-suite admission gate;
hosted CI runs pytest directly.
