# OpenPinch

Pinch analysis and total site integration toolkit: process and site heat
integration targets, multiperiod analysis, heat pumps and refrigeration,
cogeneration, utility placement, exergy, and heat exchanger network synthesis.
Python package published to PyPI; documentation on Read the Docs.

## Discipline

- Strict TDD, enforced by probity (`probity.config.ts`): failing test first,
  minimum implementation to pass, refactor on green, no speculative code.
- A test and the implementation that makes it pass go in the same commit. Run
  `pytest` on the tests covering the change after the last edit and get them
  passing before committing (probity blocks `git commit` otherwise).
- Do not suppress lint or type errors (no suppression comments), and do not
  skip or xfail tests. `skipif` with a concrete environment condition (for
  example, a solver binary or optional extra missing) is allowed.
- Every user-visible change gets an entry in `docs/release-notes.rst` under
  "Unreleased".

## Layout

- `OpenPinch/` - the package.
  - `application/` - `PinchProblem`, `PinchWorkspace` and their accessors
    (`target`, `design`, `plot`, `components`).
  - `contracts/` - Pydantic input/output contracts.
  - `domain/` - streams, zones, values and configuration fields.
  - `analysis/` - targeting, heat pumps, ORC, power, exergy, utility
    placement and heat exchanger network synthesis.
  - `optimisation/` - solver-neutral optimiser backends.
  - `adapters/`, `presentation/` - IO, plots, reports and exports.
  - `tutorials/` - packaged notebooks and sample cases.
- `tests/` - pytest suite mirroring the package layout.
- `docs/` - Sphinx documentation, including `release-notes.rst`.
- `scripts/` - CI, release and toolchain helpers (`scripts/ci_local.sh` runs
  most CI lanes locally; `scripts/pyodide` runs the suite in Pyodide).

## Commands

- Install: `uv sync --frozen --group dev` (add `--extra tespy` for TESPy).
- Unit tests:
  `uv run --no-sync pytest -m "not solver and not tespy and not performance and not docs" --hypothesis-seed=20260715`
- Focused tests: `uv run --no-sync pytest tests/<area>`
- Lint: `uv run --no-sync ruff check . && uv run --no-sync ruff format --check .`
- CI lanes locally: `scripts/ci_local.sh` (`quick`, `all`, or lane names).
  It does not run the Pyodide lane, `floors`, the py312/py313 unit lanes,
  the non-core and non-Linux smoke tests, or the dependency audit. A Pyodide
  shard runs with Node 22: `uv build --out-dir dist`, then
  `node scripts/pyodide/run_tests.mjs dist/*.whl <pytest args>` (shard paths
  and deselections are in `.github/workflows/ci.yml`; run `npm ci` in
  `scripts/pyodide` once).
- Hypothesis settings come from the `openpinch` profile in
  `tests/conftest.py`, so `--hypothesis-seed` behaves the same everywhere.
- Probity tooling (once per clone): `nvm use && npm install`, and be logged in
  to Claude Code (`claude /login`) so the TDD validator can run.

## Workflow

- Branch from `develop`; open pull requests against `develop`. CI must pass
  ("CI OK") before merging.
- Merging into `develop` bumps the patch version automatically
  (`.github/workflows/bump-version.yml`) when `develop` matches `main`.
- A release is a pull request from `develop` into `main`; merging it publishes
  to PyPI. Afterwards fast-forward `develop` to `main` so the next bump runs.

## Conventions

- Python `>=3.12`; development uses the version in `.python-version`.
- Values carry units (`{"value": ..., "unit": ...}`); multiperiod values use
  `{"values": [...], "unit": ...}`. Configuration keys are defined in
  `OpenPinch/domain/configuration_fields.py`.
- Numerical tolerances are explicit in tests; do not loosen one to make a
  test pass without recording why.
- No emojis in documentation.
