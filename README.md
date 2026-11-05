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

`manifest.json` uses `format: dinkster.base/1`. Hashes are bare lowercase sha256
hex, except `id` which has a `sha256:` prefix. Sizes are byte counts. New plans
include the display metadata below; older published manifests remain valid
without it, and unchanged builds retain their original manifest metadata.

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
| label | Display name: UTC date-pyPython-torchVersion[-cuFamily]-hash8 |
| builtAt | UTC ISO 8601 planning timestamp, retained during archive reconstruction |
| torchVersion | Exact selected torch wheel version, including any CPU/CUDA local suffix |
| torchIndex | Exact registry URL of the selected torch wheel |
| lockSource | Object with repository URL and source checkout commit used for the engine lock |

For example, a CUDA label is
`2026-10-30-py3.14.10-torch2.14.1-cu132-3f9a1c2e`; CPU and Apple Silicon
labels omit the CUDA family. The final eight characters come from the full
canonical base id, never from a metadata hash. These fields are descriptive,
not verification or cache keys. The CLI records the actual source checkout
HEAD for a new plan, not an unrelated newer engine commit. A skipped build
does not refresh its date, label or source provenance.

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
newline. `id`, `archive.parts`, `signature`, `label`, `builtAt`, `torchVersion`,
`torchIndex` and `lockSource` are excluded. Archive metadata cannot
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

New GitHub release tags are `base-<variant>-<label>`. Mirror paths are
`base/<variant>/<label>/`, containing the same named files, where label is
`YYYY-MM-DD-py<python>-torch<torch>[-cu<family>]-<hash8>`. The full canonical
id remains in the manifest. Hash-only publications remain readable until
selected by channel-aware retention. Per variant, retention keeps every
channel target, its configured predecessor by publication order, and all
newer bases pending channel adoption. Labelled and hash-only copies with the
same full id count as one generation. A hash-only alias is removable only
after the channel uses a verified labelled copy and a fresh channel install
succeeds. Stable publications, unknown directories and unmatched tags stay.

`base/<variant>/index.json` is regenerated from the paginated bucket inventory
after every mirror convergence, including skipped builds. Each row contains
`label`, full `id`, `builtAt`, `python`, `torchVersion`, `torchIndex`, total
publication `size` in bytes and bucket-relative `path`. Rows are newest first,
with legacy undated entries last; legacy display fields may be null. Size
includes manifest, extras and declared archive parts. A manifest without matching extras
is an incomplete publication and is omitted. New mirrors upload the manifest
first and extras last so interrupted transfers converge before indexing.
Retained seven-zip manifests with top-level `parts` are verified using their
historical identity fields and included with their declared volume sizes.
This metadata inventory does not restore seven-zip builds or installation.
The index is discovery metadata, never a verification or local cache key.

Before any GitHub upload and again before mirror writes, the publisher lists
the variant's existing bucket directories and verifies their manifests. An
existing hash8 with a different full id, or without an identifiable manifest,
fails publication without uploading. A retry with the same full id is allowed.
Missing R2 configuration or failed inventory reads refuse new publication
rather than bypassing collision detection.

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

To reconstruct a tar+zstd base from a saved manifest, run `build` with its
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
hosted Windows, and mac-arm64 on hosted Apple Silicon macos-15;
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
HEAD failures other than 404 fail closed. A complete mirror with a current
index performs no writes; a missing or stale index is replaced without
overwriting existing immutable release assets.
After mirror convergence, `publication.py relabel-channels` republishes any
hash-only base selected by main or stable under a readable label. It verifies
the original GitHub assets, recovers the lock source from that release's
`variants.toml`, and adds descriptive metadata without changing the content id
or archive bytes. `builtAt` is the UTC metadata preparation time, not an
invented original build date. Archive parts are copied within R2 using managed
multipart transfers with 64 MiB threshold/chunks and concurrency 4, then read
back to verify their full size and SHA256; the same verified parts are uploaded to
the labelled GitHub release. Partial attempts reuse their prepared label.
Only after verification and index regeneration does it replace the selected
base location in each channel. Conditional writes retry against the current
channel document, preserving concurrent engine and other-variant changes.
This path does not prune or remove the old publications. Consumers continue
to verify full content identities; labels do not enter local store paths.
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
origin, not the S3 API endpoint. Mirroring and retention use the locked boto3 SDK,
classic managed multipart transfers with 64 MiB threshold/chunks and concurrency
4, and standard retries bounded to 4 total attempts per request with 120s
connect/read timeouts. This repairs partial mirrors when rebuilding skips,
without an identity change or manual workflow run. An independent mirror-only
invocation without configuration skips R2 with a log message; new publication
requires configuration for its collision check. Configured errors fail rather
than claim health.
The hosted publication runner needs `gh` for GitHub release and tag operations.

One retention job runs after every successful publication matrix. Both
repositories serialize publication, channel changes and retention through the
same conditional R2 lock; a lock held by another writer is never stolen.
`retention.json` configures engine and tools newest-release counts, base
predecessors and verified aliases. Tools retention keeps the newest 10 complete
`tools/<publication-label>` directories plus every Desktop-pinned or
channel-pointed tools publication. A tools manifest is `dinkster.tools/1` with
three platform ZIP rows, and its directory is the manifest label plus the first
eight hexadecimal characters of the manifest SHA-256. `tools/index.json` binds
each directory to its publication label, manifest hash, source commit, version,
build time and total publication size. Channel tools references use the configured HTTPS public mirror
and must name the matching directory and manifest hash. `retention.py --dry-run`
lists the complete authenticated
inventory, retained targets/predecessors/pending/aliases, candidates with sizes,
unclassified directories and reclaimable bytes without deleting anything.
The workflow saves this report as the `release-retention` artifact. Application
requires recorded deletion approval and a posted dry-run listing in the config;
until both exist, the workflow only reports. Application deletes each selected
directory and its exact manifest-matched GitHub release/tag, updates indexes
and reports the remaining bucket bytes. A journal resumes partial failures
only while objects and channel protections still match. Engine pruning runs
in the engine workflow; base pruning runs here. Tools use their own `tools`
scope, index and pending journal, so engine application never selects tools
directories and tools application never selects engine releases. Tools deletion
remains disabled until issue 633 has recorded its own approval and dry-run
listing; the existing engine/base approval is not reused. There is no independent
archive-count or GitHub-only pruner. `--old-naming` additionally selects
hash-only directories and engine labels containing the full source hash,
even within the newest-release or base predecessor/pending windows. Channels,
stable releases, consumer pins and incomplete/unverified publications remain
protected. Old hash-named base releases with an empty bucket prefix are
selected only after authenticating their release manifest and content identity;
the same journal removes their release and tag without claiming bucket bytes.

Retention also reads the current Desktop tools pin and Dinkster base content
pins from their public main branches. `consumer-pins.json` is the public
producer-read reference for release selections from private consumers. It
also records unavailable inputs as `unresolved-deleted`, without pretending
that the missing publication is safe to ignore.
Consumers of independently published registry artifacts are recorded under
`nonReleaseConsumers` with their landed lock commit, path, digest and source.
The lock owns the exact pack identities; these artifacts have no engine or
base release dependency and are outside this retention policy. This provenance
is included in reports and must explicitly declare both dependencies false.
Matching release directories and their tags are retained, including tools and
pack assets outside the newest-main window. Base content pins use
the same labelled-publication preference as Dinkster's base acquisition;
verified hash aliases remain eligible when a labelled copy is available.
Unreadable or malformed sources refuse planning. Unresolved pins retain
every publication and produce a dry-run report naming each blocked input;
application refuses that report, including journal retries. Reports record
the source digests and resolved pins; application rereads them before any
deletion and before each journal entry, refusing changed inputs.
`engineDeletionEnabled` must be explicitly true before engine deletion can
run; false or missing keeps the engine CLI in dry-run mode. Enable it only
after comparing the current listing with every current consumer pin.
New release-backed consumers must register their source with retention
before relying on a publication.
Git source and Actions artifact pins are not pruned by this retention policy.

## Development gates

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright
PATH=/home/kosin/.local/bin:$PATH /home/kosin/comfy-vibe-station/run_counted_suite.sh uv run --locked python -m pytest -q
```

The counted launcher is the workspace's local full-suite admission gate;
hosted CI runs pytest directly.
