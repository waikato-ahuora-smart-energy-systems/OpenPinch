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
