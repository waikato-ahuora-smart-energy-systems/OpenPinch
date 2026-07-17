# Integration Test Instructions

## Solver-Marked HEN Matrix

```bash
uv run pytest -q -m synthesis
```

Quality-audit result: 5 passed. The segmented cases cover APOPT PDM pinch
clipping, TDM parent heat coordinates, isothermal and non-isothermal branches,
post-solve total-cost verification, process/utility segment area contributions,
and the IPOPT solve-or-actionable-rejection path.

## Optional Surface Smoke Checks

```bash
uv run python scripts/optional_install_smoke.py notebook
uv run python scripts/optional_install_smoke.py synthesis
```

Notebook/resource structure tests passed (15 tests), and a structured
segmented-input targeting example passed. The unrelated first-solve notebook
edit made during initial verification was removed during the scope audit.

## Documentation and Packaging

```bash
uv run sphinx-build -W -b html docs docs/_build/html
uv run python scripts/build_dist.py --output-dir /tmp/openpinch-dist
```

Both the warning-free documentation build and wheel/source-distribution build passed.

## Segmented Utility Cost Integration

The segmented synthesis module exercises exact ordered utility costs for partial
segment use, exact boundary use, boundary crossing, complete profile use, hot
and cold utilities, local exchanger area slices, and the IPOPT active-segment
solve-or-guidance path. It is included in the complete non-solver and
solver-marked acceptance commands above.

The complete notebook/resource, direct and indirect integration, HPR, packaging,
and documentation tests are included in the 1,978-test non-solver result.

## GitHub CI Heat-Pump Zero-Duty Follow-Up

### Heat-Pump and Segmented-Profile Integration

```bash
uv run pytest -q --hypothesis-seed=20260715 \
  tests/test_classes/test_simple_heat_pump_cycle.py \
  tests/test_classes/test_cascade_heat_pump_cycle.py \
  tests/test_classes/test_stream_segments.py
```

Verified result: 79 passed in 4.81 seconds. This covers process-duty
classification, cascade collection union behavior, positive and negative
duties, and strict segmented-profile invariants.

## Residual Compatibility Shim Removal

The integration gate covers four boundaries:

- HPR cascade orchestration passes `period_idx` once to each helper and
  propagates helper failures unchanged.
- Parsed and backend HPR records flow through cycle targeting as attribute-only
  Pydantic records.
- Exact HPR optimiser identifiers cross the HPR-to-generic-optimisation adapter
  without aliases.
- Current `StreamCollection` state round-trips through `pickle` with period and
  numeric-view behaviour intact.

Run the architecture and sole protected-contract suites with the HPR owners:

```bash
uv run pytest -q \
  tests/analysis/heat_pumps \
  tests/architecture \
  tests/e2e/test_main.py
```

No external endpoint or cleanup step is required. The tests operate on local
fixtures and temporary paths only. The protected main suite includes all four
canonical optimiser configuration values.
