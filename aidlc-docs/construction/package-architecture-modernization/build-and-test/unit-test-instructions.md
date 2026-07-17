# Package Architecture Modernization Unit Test Instructions

## Complete Non-Solver and Coverage Gate

```bash
uv run coverage erase
uv run coverage run --branch --source=OpenPinch -m pytest -m 'not solver' --hypothesis-seed=20260715
uv run coverage report --fail-under=95
```

Expected final result: 2,039 passed, 4 solver tests deselected, no unexpected
warnings, 97.95 percent statement coverage, and 92.79 percent branch coverage.

## Supported Solver Gate

```bash
uv run pytest -q -m solver --hypothesis-seed=20260715
```

Expected final result: 3 passed and the explicitly excluded nine-stream live
benchmark skipped.

## Owner-Focused Gates

```bash
uv run pytest -q tests/domain tests/contracts tests/optimisation
uv run pytest -q tests/application tests/analysis tests/adapters tests/presentation
uv run pytest -q tests/architecture tests/packaging tests/e2e
```

These groups protect domain outcomes, wire contracts, generic optimisation,
orchestration, engineering calculations, infrastructure translation,
presentation data, dependency directions, artifacts, and the sole external
contract.

## Property-Based Testing

Always include `--hypothesis-seed=20260715`. Do not disable shrinking. Generated
tests cover round trips, segment transactions, interval insertion, ownership,
ordering, conservation, candidate ranking, and reproducibility.
