# Dinkster-Environments

Content-addressed Python base bundles for Dinkster hardware variants. Engine
releases and environment installation belong to Dinkster, not this repository.
Clients never resolve dependencies. The publisher uses uv to select binary
variant wheels at the settled versions, constraining every applicable registry
entry to the named Dinkster commit's `uv.lock`. Existing locked artifact hashes
and sizes cannot be overridden by index responses. Variant torch hashes come
from their torch index, not from the PyPI torch artifact in the lock.

## Variants

`variants.toml` pins Python 3.14 and python-build-standalone, torch 2.14.1,
numpy 2.5.1, opencv-contrib-python-headless 5.0.0.93 and torchvision 0.29.1.
Before moving torchvision, check cp314 wheels on every configured torch index,
not just PyPI. The dependency closure
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
| platform | Object with os (windows or linux) and arch (x86_64) |
| python | Full interpreter version |
| interpreter | Object with release, filename, url, size, sha256 and executable |
| wheels | Filename-sorted array of name, version, filename, size, sha256, index, url, verification, lockedVersion |
| replaces | Sorted unique normalized distribution names replaced in engine constraints |
| workerProtocol | Positive integer from the named engine commit's workers source |
| nvidia | null for CPU, otherwise minimumDriver, minimumComputeCapability, maximumExclusiveComputeCapability |
| parts | Ordered array of filename, size, sha256 for split archive volumes |
| signature | null; reserved for ed25519 signing |

NVIDIA bounds are decimal strings; the maximum is null when unbounded.
Driver and compute capability values are compared as numeric dotted components,
not lexicographically or as floating point. `platform` is the OS/architecture
discriminator; clients do not infer hardware rules from variant names.
`interpreter.release` is the python-build-standalone release. `url` is the exact
artifact URL, not an index search. Each wheel's `index` is its source registry.
Keeping both makes manifest-only reconstruction independent of
an index's current resolution. A removed upstream artifact fails verification;
the builder never substitutes another wheel.

`verification` is `lock` when the exact artifact's hash and size match Dinkster's
lock, otherwise `index`. `lockedVersion` records the corresponding Dinkster lock
version, or null when absent. A CPU/CUDA local version such as `2.14.1+cpu`
matches the locked public version `2.14.1` but does not inherit its PyPI wheel
hash. Until Dinkster's lock carries all base pins, missing entries are selected
only at the exact settled root versions and remain explicitly index-verified.
Every wheel's provenance is printed in planning output and bound by base id.
Download verification checks all recorded hashes regardless of provenance.

`wheels` is the selected variant's install set. `replaces` is its authoritative
replacement set: selected wheel names plus the universal Dinkster-lock dependency
closure of the base roots. It includes all platform-marker branches and follows
only requested dependency extras, including torch's cuda-toolkit extras. Thus a
CPU base replaces CUDA-only engine-export rows without installing CUDA wheels.
The assembler removes engine constraint rows named in `replaces`, then appends
the exact requirements/hashes from `wheels`. A replacement name absent from
`wheels` means remove-only, not fetch or install. Other engine constraints remain
unchanged, and selected-wheel versions must still match engine lock versions.
Names use Python distribution normalization (lowercase, runs of hyphens,
underscores and dots become one hyphen), never prefix-based inference.

Identity is sha256 of the UTF-8 JSON object containing exactly `format`,
`variant`, `platform`, `python`, `interpreter`, `wheels`, `replaces`,
`workerProtocol` and `nvidia`,
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

Consumer planning fixtures are in `tests/fixtures/base/<variant>.json` and use
this complete manifest format, including real index metadata and empty `parts`
because they are plans, not published downloadable archives. They require no
fixture-specific decoder. Installation tests must build archive parts first.
Fixture lock: Dinkster engine-base commit
`44eb74161881f0eb33fd62ed8b46ccdc4de50754` (pushed 2026-10-28 12:01 PDT).
This explicit fixture input is separate from the workflow source pin in
`variants.toml`.

## Local build and reconstruction

Requires uv 0.11.29 and 7-Zip (`7z` or `7zz`) on PATH. The Dinkster source checkout must
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
It checks locked public versions and variant metadata before consulting indexes,
retaining the previous manifest and id when inputs match. In particular, torch
and torchvision compare locked versions and the variant index, never PyPI lock
hashes against CPU/CUDA wheel hashes. Other already-locked base artifacts must
still match their locked hashes and sizes. A changed replacement-name closure
also requires a new base. A missing entry temporarily uses the
settled variant pin under the documented index-only provenance rule.

## Hosted builds and publication

Branch pushes build all four variants on hosted Linux runners; cross-platform
wheel selection never executes the target interpreter. Each new base is rebuilt
from its manifest and uploaded as a workflow artifact. No hosted workflow needs
to be dispatched manually.

Landing build-input changes on main is the publication gate while the program
is being established. After successful builds, the publish job creates the
content-addressed GitHub releases itself, then optionally mirrors to R2.
Unchanged bases skip building and publication. The publisher checks the
repository identity and main ref; it never publishes from a feature branch.
The publish job references the `base-publication` environment, which initially
has no protection rules and does not require approval. Once the publication
path is stable, the user intends to become the approval gate again: add a
required reviewer to `base-publication` in the repository environment settings.
That enables the approval step without a code change.

R2 needs `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` repository secrets, plus
`R2_ACCOUNT_ID`, `R2_BUCKET` and `R2_PUBLIC_BASE_URL` variables, using the same
names as Dinkster and Dinkster-Desktop. The S3 endpoint is derived as
`https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com`; the public URL is the serving
origin, not the S3 API endpoint. Credentials are mapped to the AWS CLI only in
its subprocess environment. Missing any of these skips R2 with a log message;
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
