# Dinkster-Environments

Content-addressed Python base bundles for Dinkster hardware variants. Engine
releases and environment installation belong to Dinkster, not this repository.
Clients never resolve dependencies. The publisher uses uv to select binary
variant wheels at the settled root versions. It constrains interface names to
the named Dinkster commit's `uv.lock`: base-owned distributions that non-base
engine packages depend on for the target, computed from dependency edges rather
than a static list. Base-only closure names resolve from the variant indexes,
not the generic development lock's incompatible CUDA closure. Every selected
artifact carries its exact URL, size and hash; clients never substitute wheels.

## Variants

`variants.toml` pins Python 3.14 and python-build-standalone, torch 2.14.1,
numpy 2.5.1, opencv-contrib-python-headless 5.0.0.93 and torchvision 0.29.1.
Before moving torchvision, check cp314 wheels on every configured torch index,
not just PyPI. The dependency closure
of these packages belongs to the base; all other wheels belong to the engine.

Windows keeps CPython 3.14.10 from python-build-standalone 20261026, whose
separate `DLLs/sqlite3.dll` contains SQLite 3.53.1. The builder replaces only
that DLL with SQLite 3.53.1 from the hash-pinned official amalgamation plus
the [upstream mapped-shm recovery fix](https://sqlite.org/src/info/7c103a0b7d).
Zig 0.14.1 compiles the x86_64 Windows DLL; compiler flags, patched source
hash and DLL hash/size are pinned under `interpreters.windows.sqlite` and
carried into `interpreter.sqlite` in each Windows base manifest. A changed
override changes the base identity. Linux's interpreter remains unchanged.
Zig is a new build-only dependency in the `sqlite-build` group. Native Windows
CI verifies the exact Zig wheel download size and SHA-256, compares the
installed compiler files to that wheel, and reproduces the pinned DLL hash.
The manifest also records that toolchain artifact and the SHA-256 of the
applied unified `os_win.c` changes in the amalgamation (`sqlite.patch`).

The native Windows builder and archive verifier both require persisted WAL
reads and `integrity_check=ok` while an old 64 KiB `-shm` mapping remains
open, then again after release. This detects the
[Windows regression](https://sqlite.org/forum/forumpost/f9c68d01e2d7f638)
without rejecting fixed builds of the same SQLite version. No application
retry or database journal-mode change is involved. Drop the override when
the pinned interpreter includes an upstream release with the fix; retain
the functional gate.

| Variant | Platform | Torch index | NVIDIA bounds |
| --- | --- | --- | --- |
| win-nvidia | Windows x64 | cu132 | driver >=580, capability >=7.5 |
| win-nvidia-cu126 | Windows x64 | cu126 | driver >=525, capability >=5.0 and <10.0 |
| win-cpu | Windows x64 | cpu | none |
| linux-cpu | Linux x64, glibc >=2.28 | cpu | none |
| linux-nvidia | Linux x64, glibc >=2.28 | cu132 | driver >=580, capability >=7.5 |
| mac-arm64 | macOS 14 or newer, Apple Silicon arm64 | PyPI (MPS) | none |

Intel macOS is unsupported. Apple Silicon uses the default PyPI torch wheels,
not a separate torch index or CUDA dependencies.

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
| archive | Object with format (tar+zstd) and ordered parts (filename, size, sha256) |
| variant | Variant name from variants.toml |
| platform | Object with os (windows, linux or macos) and arch (x86_64 or arm64) |
| python | Full interpreter version |
| interpreter | Object with release, filename, url, size, sha256 and executable |
| wheels | Filename-sorted array of name, version, filename, size, sha256, index, url, verification, lockedVersion |
| replaces | Sorted unique normalized distribution names replaced in engine constraints |
| workerProtocol | Positive integer from the named engine commit's workers source |
| nvidia | null for CPU, otherwise minimumDriver, minimumComputeCapability, maximumExclusiveComputeCapability |
| signature | null; reserved for ed25519 signing |

NVIDIA bounds are decimal strings; the maximum is null when unbounded.
Driver and compute capability values are compared as numeric dotted components,
not lexicographically or as floating point. `platform` is the OS/architecture
discriminator; clients do not infer hardware rules from variant names.
`interpreter.release` is the python-build-standalone release. `url` is the exact
artifact URL, not an index search. Each wheel's `index` is its source registry.
Windows also carries `interpreter.sqlite`: official source URL/hash/size,
upstream fix URL, compiler and flags, patched source hash, installed path,
DLL filename/hash/size, and the unchanged SQLite version. Windows builds
require the native target OS and the `sqlite-build` dependency group.
Keeping both makes manifest-only reconstruction independent of
an index's current resolution. A removed upstream artifact fails verification;
the builder never substitutes another wheel.

`verification` is `lock` when the exact artifact's hash and size match Dinkster's
lock, otherwise `index`. `lockedVersion` records the corresponding Dinkster lock
version, or null when absent. A CPU/CUDA local version such as `2.14.1+cpu`
matches the locked public version `2.14.1` but does not inherit its PyPI wheel
hash. Base-only names may have versions different from the development lock;
those artifacts remain explicitly index-verified, with `lockedVersion` recording
the development version for provenance, not an installation requirement.
Interface versions must match the target lock, and an index cannot override
the hash/size of an existing interface artifact with the same filename.
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
unchanged. Only interface-name versions must match the engine development lock;
base-only dependencies follow the chosen variant. Interface edges include
target-applicable normal and optional dependencies of non-base packages.
Names use Python distribution normalization (lowercase, runs of hyphens,
underscores and dots become one hyphen), never prefix-based inference.
The producer's per-variant `replaces` list is the single ownership-by-name
contract for engine builders and installers; they do not maintain another list
or infer ownership from byte-identical wheels shared across bases.

Identity is sha256 of the UTF-8 JSON object containing exactly `format`,
`archive`, `variant`, `platform`, `python`, `interpreter`, `wheels`, `replaces`,
`workerProtocol` and `nvidia`. The identity's `archive` object contains only
`{"format":"tar+zstd"}`, not `parts`. Serialize
with sorted object keys, compact separators, ASCII escaping, and no trailing
newline. `id`, `archive.parts` and `signature` are excluded. Archive metadata cannot
create a circular content address; rebuilding may yield different archive part
hashes while retaining the same verified payload identity. Engine-only lock
changes do not affect identity. Interpreter, target wheel, protocol or hardware
metadata changes do.

The archive is a single PAX tar stream compressed with standard zstd at level 3
using `zstandard.ZstdCompressor(threads=-1)`, then split by bytes. Concatenating
`archive.parts` in manifest order reconstructs the complete zstd stream;
there is no per-part framing. Relative paths, directory and executable modes,
and internal symlinks are preserved. Absolute paths, parent traversal, escaping
symlinks, hard links and device nodes are refused. Both producer validation and
native extraction gates use `ZstdDecompressor().stream_reader`, streaming
`tarfile` extraction and the stdlib data filter plus these contract checks.
The archive format participates in identity independently of `workerProtocol`,
which remains the protocol declared by the named engine source. Legacy 7z
manifests require a new build; existing releases are not overwritten.

The tar archive contains `python/` and `wheels/`. The executable is
`python/python.exe` on Windows or `python/bin/python3` on Linux and macOS, relative to
the extracted root. Parts are `base.tar.zst.001` onwards, each at most
1,900,000,000 bytes and strictly below
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
this complete manifest format, including real index metadata and empty `archive.parts`
because they are plans, not published downloadable archives. They require no
fixture-specific decoder. Installation tests must build archive parts first.
Fixture lock: Dinkster engine-base commit
`5b84da719dc5888b9b3d1dfd7213e4841001b5a0` (landed 2026-10-29 05:44 PDT),
the same source pin as `variants.toml`.

`tests/fixtures/linux-nvidia-resolved.json` records the publisher resolver's
cu132/Python 3.14.10 binary selection at the exact base root pins, without
engine constraints. It covers all 33 Linux base distributions, including
manylinux1/2010 CUDA wheels that are valid on the glibc 2.28 target. The closure
test feeds these fixed artifacts through the planner and asserts the complete
install and ownership sets. It does not assert compatibility with the pinned
engine lock and is not a downloadable base.

## Local build and reconstruction

Requires uv 0.11.29; `uv sync` installs the locked zstandard package. No external
archiver is needed. The Dinkster source checkout must
be at the intended commit; no engine imports or engine dependencies are needed.

```sh
uv sync --locked
uv run --locked python bundles.py plan --source ../Dinkster --variant linux-cpu --output plan.json
uv run --locked python bundles.py build --manifest plan.json --output dist/linux-cpu
uv run --locked python bundles.py verify --directory dist/linux-cpu
```

To reconstruct a tar+zstd base removed by retention, run `build` with its
`manifest.json` instead of `plan.json`. Only that manifest and the upstream
interpreter and wheel artifacts are needed. Its id and every download's size
and sha256 are checked before archiving. The output directory must be empty.
For incremental builds, `plan --previous <published-manifest>` prints an explicit
skip reason and writes `skip=true` to GITHUB_OUTPUT when identity is unchanged.
It checks root pins, interface versions and variant metadata before consulting
indexes, retaining the previous manifest and id when inputs match. Torch and
torchvision compare their root pins and variant index, never PyPI lock hashes
against CPU/CUDA wheel hashes. Interface artifacts retain their lock hash/size
checks. Development-lock changes to base-only versions do not republish a base;
their exact published selection remains authoritative. A changed replacement-name
closure also requires a new base. Protocol, interpreter, archive and hardware
metadata remain compatibility inputs.

## Hosted builds and publication

Branch pushes build Linux variants on hosted Linux and Windows variants on
hosted Windows, and mac-arm64 on hosted Apple Silicon macos-14;
wheel selection remains explicit. Each new base is rebuilt
from its manifest and uploaded as a workflow artifact. A native target OS
verification job decodes every produced base with Python tarfile and zstandard,
checks member safety and wheel hashes, and runs its executable interpreter with
`-I -c "import sys, sysconfig"` without chmod. Publication requires all native
verification jobs to pass. Consumer-side Windows, Linux and macOS behavior is
owned by Dinkster; the producer's native gates cover all configured variants.
No hosted workflow needs to be dispatched manually.

Landing build-input changes on main is the publication gate while the program
is being established. After successful builds, the publish job creates the
content-addressed GitHub releases itself. A new variant or a change to Python,
torch's variant index/family, or another base-owned distribution pin requires a
new base. Engine releases do not trigger base publication. In particular,
comfy-kitchen and comfy-aimdo are engine-side dependencies, not base-owned pins;
changing them does not rebuild or publish a base. Protocol, archive and hardware
metadata remain part of compatibility and content identity.
Unchanged bases skip rebuilding and
GitHub publication, but every main publication runs R2 convergence for each
variant's newest release. It verifies GitHub metadata digests and canonical
manifest identity, HEADs every expected R2 object, and downloads only missing
assets from that exact release. Downloaded parts must match both release and
manifest sizes/SHA256 before upload. Present objects are never overwritten;
HEAD failures other than 404 fail closed. A complete mirror performs no writes.
The publisher checks the
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
its subprocess environment for retention. Mirroring uses the locked boto3 SDK,
classic managed multipart transfers with 64 MiB threshold/chunks and concurrency
4, and standard retries bounded to 4 total attempts per request with 120s
connect/read timeouts. This repairs partial mirrors when rebuilding skips,
without an identity change or manual workflow run. Missing configuration skips
R2 with a log message; configured errors fail rather than claim health.
The hosted publication runner needs `gh` and the AWS CLI (retention only).

## Development gates

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright
PATH=/home/kosin/.local/bin:$PATH /home/kosin/comfy-vibe-station/run_counted_suite.sh uv run --locked python -m pytest -q
```

The counted launcher is the workspace's local full-suite admission gate;
hosted CI runs pytest directly.
