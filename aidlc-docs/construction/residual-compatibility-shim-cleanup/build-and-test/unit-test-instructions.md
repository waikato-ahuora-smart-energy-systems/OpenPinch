# Unit Test Instructions

## Focused Contract Tests

Run the numerical, HPR, unit, input, HEN backend, comparison-tool, architecture, documentation, and workbook tests:

```bash
uv run pytest -q tests/analysis/test_miscellaneous.py tests/analysis/test_support_methods.py tests/analysis/heat_pumps/test_optimisation_adapter.py tests/analysis/heat_pumps/test_shared.py tests/contracts/test_unit_system.py tests/contracts/test_input.py tests/contracts/test_heat_exchanger_network_input.py tests/analysis/heat_exchanger_networks/test_pinch_design_method.py tests/packaging/test_openhens_comparison_prerequisite.py tests/architecture/test_compatibility_shims.py tests/architecture/test_api_boundary.py tests/architecture/test_package_markers.py tests/packaging/test_docs_consistency.py tests/adapters/test_workbook.py
```

Expected result for the implemented revision: 275 passed.

## Complete Non-Solver Suite

```bash
uv run pytest -q -m "not solver" --hypothesis-seed=20260715
```

Expected result for the implemented revision: 2,108 passed, 3 intentional opt-in skips, and 4 solver deselections.

## Property-Based Tests

The penalty properties use bounded finite residuals, both `PenaltyForm` members, Hypothesis shrinking, and seed `20260715`. Failures must be reproduced from the shrunk example before changing numerical equations.

## Static Quality Gates

```bash
uv run ruff check .
uv run ruff format --check .
git diff --check
```

All three commands must pass without findings.
