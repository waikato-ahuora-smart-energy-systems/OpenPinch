# HPR and MVR Notebook Reliability End-to-End Test Instructions

## Real Tutorial Workflow

Run every generated HPR/MVR notebook as a separate pytest case:

```bash
OPENPINCH_TUTORIAL_PROFILES=slow-hpr \
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/packaging/test_notebooks.py::test_slow_hpr_notebook_executes
```

The four cases are:

1. Carnot heat pump and refrigeration.
2. CoolProp vapour-compression heat pump and refrigeration, with optional
   Brayton and TESPy evidence.
3. Five required shared-design multiperiod configurations plus one optional
   advanced cascade diagnostic.
4. Direct process MVR plus the required bounded VC+MVR cascade.

Success requires all four notebook cases to execute every code cell without an
unexpected exception. Required paths must solve directly. Optional paths may
return only their documented typed status records.

## Standard Benchmark Workflow

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/e2e/test_hpr_mvr.py
```

This runs 54 distinct standard-corpus HPR assignments plus one separate direct
process-MVR workflow. The matrix covers 18 VC heat-pump, 18 VC refrigeration,
and 18 VC+MVR assignments across process- and site-level integration.

## Expected Evidence

- Finite search budgets and bounded public observations.
- Atomic public result publication with no partial failed target.
- Valid solution evidence where a solve is required.
- Strict improvement toward the best viable observed objective for designated
  convergence sentinels; no claim of global optimality.
- Notebook and code-cell attribution for any failure.
