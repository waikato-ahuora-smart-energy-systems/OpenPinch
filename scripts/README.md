# Repository scripts

## Overview
CI, release, toolchain and fixture-generation helpers. They are not part of the `OpenPinch`
package and sit outside its namespace; there is no `scripts/__init__.py`, and tests import some of
them as `scripts.<name>`. Most are invoked by the GitHub workflows in `.github/workflows` or by a
developer from the repository root. The scripts that import `OpenPinch` need an environment where
the package is installed.

## Interface
CI and release
- `ci_local.sh` (`ci_local.sh`): bash. Runs CI lanes locally (`lint unit docs tespy performance
  solver notebooks build`, plus `quick` and `all`; `--linux` reruns inside a Docker container).
  Called by `.githooks/pre-push`; documented in `README.md` and `AGENTS.md`.
- `check_lockfile_version.py` (`check_lockfile_version.py`): exits 1 when the editable `openpinch`
  entry in `uv.lock` differs from `pyproject.toml`. Used by `ci.yml` and `bump-version.yml`, and by
  the `lint` lane of `ci_local.sh`.
- `check_release_version.py` (`check_release_version.py`): `read_project_version`,
  `validate_version_advance`; prints a strict `X.Y.Z` version, optionally requiring it to exceed
  `--base-pyproject`. Used by `ci.yml`, `release.yml`, `bump-version.yml`.
- `release_plan.py` (`release_plan.py`): `decide`, `parse`, `expected_files`, `UnsafeRelease`;
  prints `publish=true|false` from tag and release listings. Used by `release.yml`.
- `build_dist.py` (`build_dist.py`): builds sdist and wheel into `dist/` (or `--output-dir`).
- `build_docs.py` (`build_docs.py`): builds the Sphinx HTML tree with `--fail-on-warning`
  into `docs/_build/html` (or `--output-dir`).
- `artifact_install_smoke.py` (`artifact_install_smoke.py`): `is_checkout_source_import`; smoke-checks
  an installed wheel (`--surface core|tespy`). Used by `ci.yml` and `ci_local.sh`.
- `optional_install_smoke.py` (`optional_install_smoke.py`): smoke-checks an install surface
  (`core dashboard notebook brayton_cycle tespy synthesis`). Used by `ci.yml` and `ci_local.sh`.

Developer tooling
- `update_toolchain.py` (`update_toolchain.py`): runs `uv python install`, `uv python pin`,
  `uv lock --upgrade` and `uv sync` for the version in `.python-version`.
- `format_repo.py` (`format_repo.py`): runs `uv run --group dev ruff format .`.

Generators
- `generate_analysis_catalog.py` (`generate_analysis_catalog.py`): `render_catalog`, `DESTINATION`;
  writes `docs/_data/analysis-methods.csv` from `METHOD_CATALOG`.
- `generate_tutorial_coverage.py` (`generate_tutorial_coverage.py`): writes
  `docs/_data/tutorial-coverage.csv` by introspecting the public accessors and notebook metadata.
- `generate_hpr_performance_map_contract.py` (`generate_hpr_performance_map_contract.py`):
  `render_contract_resources`, `check_contract_resources`, `write_contract_resources`
  (`--check` or `--write`); owns `OpenPinch/contracts/resources/hpr_performance_map`.

OpenHENS migration and benchmarks (manual, not run in CI)
- `convert_openhens_fixtures.py` (`convert_openhens_fixtures.py`): `parse_openhens_csv`,
  `convert_case_to_target_input`, `write_migration_fixtures`, `ConversionError`; converts OpenHENS
  CSV cases to JSON fixtures (needs `--write`).
- `compare_openhens_openpinch_top5.py` (`compare_openhens_openpinch_top5.py`): runs OpenHENS and
  OpenPinch on fixture cases and writes top-5 comparison CSV and Markdown reports.
- `benchmark_performance.py` (`benchmark_performance.py`): `time_hens_tier_case`,
  `time_hens_tier_case_with_timeout`, `summarize_hens_tier_comparisons`,
  `small_hens_benchmark_cases`, `BenchmarkSolverCallTracer`, `BenchmarkTraceExecutor`; non-gating
  HEN synthesis quality-tier benchmark writing incremental JSON.

- Subdirectories: [pyodide](./pyodide/README.md): Node runner for the suite under Pyodide.

## Invariants
- Release versions must match `X.Y.Z` exactly and, with a base file, strictly exceed it
  [check_release_version.py:validate_version_advance].
- A release is skipped only when its GitHub release has both the wheel and sdist attached; when the version is not already released, a lower
  version than any existing tag, or a tag on another commit, is an error (exit 1)
  [release_plan.py:decide].
- The generators write canonical output: LF line endings and a fixed column order; HPR resources are
  sorted-key JSON, and the fixture version is pinned so golden files do not change with the
  installed release [generate_hpr_performance_map_contract.py:FIXTURE_GENERATOR_VERSION].
- `artifact_install_smoke.py` must run against the installed wheel, not the checkout
  [artifact_install_smoke.py:is_checkout_source_import].
- The core install surface must not import optional dependencies such as plotly or tespy
  [optional_install_smoke.py:_check_core_surface].
- The converter rejects OpenHENS cases whose exchange, heating and cooling economics differ, because
  OpenPinch costing has one set [convert_openhens_fixtures.py:_validate_costing_can_use_openpinch_configuration].
- The OpenHENS comparison needs an OpenHENS checkout that exposes the modules it imports
  [compare_openhens_openpinch_top5.py:_supported_openhens_checkout].
- The tutorial manifest lists the public members of the accessors; a member must be mapped to a
  tutorial or listed in `UNDER_DEVELOPMENT_METHODS`
  [generate_tutorial_coverage.py:UNDER_DEVELOPMENT_METHODS].

## Side Effects
- Subprocesses: `build_dist.py` (`python -m build` or `uv run`), `build_docs.py` (sphinx),
  `format_repo.py`, `update_toolchain.py` (uv), `artifact_install_smoke.py` (`python -m OpenPinch
  notebook --help`), `ci_local.sh` (uv, pytest, coverage, docker, git).
- `build_dist.py` deletes stale `openpinch-*` wheels, sdists and attestation files in the output
  directory before building [build_dist.py:_clean_output_dir].
- `ci_local.sh` creates `reports/`; its `build` lane removes and recreates `dist/` [ci_local.sh:lane_build]; it syncs the uv environment;
  the `solver` lane may download IDAES extensions.
- `update_toolchain.py` changes the pinned Python, `uv.lock` and the synced environment
  [update_toolchain.py:main].
- Writes repository files: `generate_analysis_catalog.py`, `generate_tutorial_coverage.py`,
  `generate_hpr_performance_map_contract.py --write`, `convert_openhens_fixtures.py --write`
  (fixtures under `tests/fixtures/openhens`, packaged sample cases and adapter snapshots).
- `compare_openhens_openpinch_top5.py` writes into `--output-dir`, temporarily edits `sys.path` and
  `sys.modules`, prepends `~/.idaes/bin` to `PATH`, uses temporary directories, and spawns
  multiprocessing workers [compare_openhens_openpinch_top5.py:_configure_solver_path].
- `benchmark_performance.py` forks worker processes, monkeypatches
  `solver_backend.solve_gekko_model` while tracing, runs solvers, and writes the JSON results file
  (default under `results/`) [benchmark_performance.py:BenchmarkSolverCallTracer.patched_backend].
- `release_plan.py` reads the tag and release files named on its command line.

## Verification
- Run: `uv run --no-sync pytest tests/packaging` (covers the build, release, entrypoint, workflow and
  Pyodide scripts; the generators and OpenHENS tools are covered by
  `tests/application/test_analysis_contracts.py`, `tests/contracts/test_hpr_performance_map.py`,
  `tests/analysis/heat_exchanger_networks/test_benchmark_performance.py` and
  `tests/analysis/heat_exchanger_networks/test_heat_exchanger_network_array_adapter.py`).
- `generate_tutorial_coverage.py` has no direct test (the packaging tests only read `docs/_data/tutorial-coverage.csv`).
- `format_repo.py` and `update_toolchain.py` have no direct test of their own beyond
  `tests/packaging/test_packaging_metadata.py` for the latter.
