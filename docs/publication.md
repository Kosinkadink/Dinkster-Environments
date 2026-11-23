# Publication setup and approval

Engine releases and installation are owned by Dinkster. The base roots,
interpreter, target indexes and hardware definitions live in `variants.toml`;
manifest and archive contracts live in `bundles.py` and its tests. Consumers
must use the producer's `replaces` list, not maintain another ownership list or
infer ownership from byte-identical wheels shared across bases. Driver and
compute capability values are numeric dotted components, not floats or
lexicographic strings; consumers must use `platform`, not infer hardware rules
from variant names.

## Hosted publication

Landing build-input changes on main is the publication gate while the program
is being established. No hosted workflow needs manual dispatch. Engine releases
alone do not require a new base. Once the publication path is stable, the user
intends to become the approval gate again: add a required reviewer to the
`base-publication` repository environment. That enables approval without a code
change. The environment initially has no protection rules.

Configure these repository secrets:

- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

Configure these repository variables, using the same names as Dinkster and
Dinkster-Desktop:

- `R2_ACCOUNT_ID`
- `R2_BUCKET`
- `R2_PUBLIC_BASE_URL`

The public URL is the serving origin, not the S3 API endpoint. The hosted
publication runner also needs `gh` for GitHub release and tag operations.
The [base workflow](../.github/workflows/bases.yml) is the executable reference
for native build/verification, mirror convergence and retention commands.

## Recovery and consumer obligations

Keep published interpreter and wheel artifacts available for manifest-only
reconstruction. A removed artifact must not be replaced with another wheel.
Legacy 7z manifests require a new build; do not overwrite existing releases.
Verify every archive part before extraction and every wheel before installing.

A hash-only alias is removable only after the channel uses a verified labelled
copy and a fresh channel install succeeds. Only the engine release workflow
holds the channel signing key; maintenance does not need it. Do not replace
signed-channel verification with the old unsigned CLI on a signature failure.
Use [maintenance](maintenance.md) for authenticated reports and cleanup gates.
