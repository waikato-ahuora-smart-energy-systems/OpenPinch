# HPR and MVR Notebook Reliability Integration Test Instructions

## Purpose

Exercise the interactions among public targeting accessors, CoolProp-backed
cycles, shared multiperiod placement, process MVR components, VC+MVR cascades,
and the standard benchmark corpus.

## Focused HPR and MVR Integration Gate

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/e2e/test_hpr_mvr.py \
  tests/analysis/heat_pumps/test_multiperiod_hpr.py \
  tests/analysis/heat_pumps/test_process_components.py \
  tests/analysis/heat_pumps/test_vapour_compression_mvr.py \
  tests/analysis/heat_pumps/test_hpr_coolprop_simulator.py \
  tests/analysis/heat_pumps/test_targeting.py
```

The verified selection contains 192 separately collected tests and must finish
with no failure.

## Integration Scenarios

### Standard benchmark to public HPR service

- Run 54 deterministic standard-problem assignments.
- Require a successful target, a legitimate no-op, or a bounded typed failure.
- Require all convergence sentinels to improve toward the best viable observed
  objective within their finite search budgets.

### Shared installed design across periods

- Build a fresh problem for each technology configuration.
- Optimize one scalar shared design across exactly `turndown`, `base`, and
  `peak`.
- Require aligned period results, finite evidence, and success for every
  required design.
- Keep independent `target.all_periods` replay distinct from shared-design
  optimization.

### Process MVR and VC+MVR

- Verify direct process-MVR component lifecycle, stage work, replacement
  streams, serial/parallel evidence, and downstream targeting.
- Verify the required one-stage VC+MVR target with explicit topology, fluids,
  and bounded search settings.

### Target to performance map

- Verify that required CoolProp VC targeting retains the winning simulation
  record and that public map generation uses target-owned evidence.

## Environment and Cleanup

No network service, database, or external solver is needed. Pytest temporary
directories isolate copied notebook execution and are cleaned by pytest.
