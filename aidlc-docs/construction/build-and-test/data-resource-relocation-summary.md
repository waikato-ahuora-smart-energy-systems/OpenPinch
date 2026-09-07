# Data resource relocation verification

Moved the three HPR contract JSON resources to
OpenPinch/contracts/resources/hpr_performance_map and the compressor
characteristic JSON to
OpenPinch/analysis/heat_pumps/performance_maps/characteristics. Removed the
obsolete OpenPinch/data package, its remaining initializer files and caches.
Resource helper APIs and JSON bytes are unchanged. Existing tutorial relocation
changes, including all 32 original tutorial file hashes, are preserved.

Updated both resource loaders, the contract generator, existing path-dependent
contract and packaging tests, and the API resource documentation. No current
code or documentation references OpenPinch/data; historical audit records are
retained.

## Build and test instructions and results

- Run .venv/bin/python scripts/generate_hpr_performance_map_contract.py --check.
  Result: passed; canonical resources match the generator.
- Run .venv/bin/python -m pytest tests/packaging
  tests/contracts/test_hpr_performance_map.py
  tests/analysis/heat_pumps/test_hpr_tespy_simulator.py with the selection
  excluding test_base_profile_notebook_executes and
  test_optional_profile_notebooks_execute, --hypothesis-seed=20260715 -q -ra.
  Result: 218 passed, 14 deselected in 64.54 seconds. Includes strict Sphinx,
  resource copying, canonical identity, contract properties and TESPy consumers.
- Run .venv/bin/python scripts/build_dist.py --output-dir with a temporary
  artifact directory. Extract the sdist and build another wheel with
  python -m build --wheel --no-isolation. Both formats built successfully.
- Install each wheel with uv pip install --no-deps --target into a separate
  temporary directory. From outside the checkout, use that target as PYTHONPATH
  and run scripts/artifact_install_smoke.py --repo-root with the checkout path
  and --surface tespy, matching the available development dependencies.
  Both installed checks passed, including target/map generation and contracts.
- All four resource hashes match before and after relocation, inside wheel and
  sdist archives, and in both installed wheels. Old data paths are absent from
  both distributions and OpenPinch.data cannot be imported in either install.
- Ruff check, formatting and git diff whitespace checks passed. Full numerical
  and notebook execution suites were not repeated for this path-only change.

Logs: /tmp/openpinch-data-resource-tests.log,
/tmp/openpinch-data-resource-build.log and
/tmp/openpinch-data-resource-installed.log. Artifacts:
/tmp/openpinch-data-resource-dist.

## Extension applicability

PBT-01: N/A to new functional design; this is a path relocation with an explicit
byte-preservation requirement. PBT-02 and PBT-03: compliant through unchanged
contract round-trip and invariant tests, plus canonical resource identity checks.
PBT-04, PBT-05 and PBT-06: N/A; no idempotent algorithm, alternative algorithm or
stateful operation introduced. PBT-07, PBT-08 and PBT-09: compliant using existing
domain strategies, Hypothesis shrinking and seed 20260715. Security and
Resiliency remain disabled and were skipped. Performance and infrastructure
testing: N/A; neither behavior nor deployment architecture changed.

Implementation and verification are complete. No commit or push performed.
