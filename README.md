# Dinkster-Environments

Content-addressed Python base bundles for Dinkster hardware variants. Engine
releases and environment installation belong to Dinkster, not this repository.

## Run locally

Install uv 0.11.29, then from this repository:

```sh
uv sync --locked
uv run --locked python bundles.py --help
uv run --locked python -m pytest -q tests/test_pypi_cleanup.py
```

To build a base, use a Dinkster source checkout at the intended commit and the
[local build procedure](docs/development.md). A full base build downloads the
interpreter and variant wheels; it is not needed to run the mocked tests.

## Reference and operations

- [Development and reconstruction](docs/development.md)
- [Publication setup and approval](docs/publication.md)
- [Retention and PyPI cleanup](docs/maintenance.md)
- [Variant and interpreter pins](variants.toml)
- [Bundle planner, archive builder and verifier](bundles.py)
- [Publication implementation](publication.py) and [contract tests](tests/test_bundles.py)
- [Retention policy](retention.json) and [tests](tests/test_retention.py)

CLI reference is available through `--help` on `bundles.py`, `publication.py`,
`signed_retention.py`, `pypi_inventory.py` and `pypi_cleanup.py`.
