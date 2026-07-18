# Integration Test Instructions

## Contract Integration Scenarios

### Penalty Selection and HPR Callers

Run the numerical and heat-pump tests to prove every production caller supplies `PenaltyForm` and both equations remain unchanged.

### Unit Configuration and Runtime Conversion

Run the unit and input contract tests. One dimensional configuration key must apply to every field in its `unit_groups` tuple while direct field overrides retain precedence.

### HEN Dependency Boundary

Run the HEN backend tests. Pyomo must receive one keyword-form availability call; an incompatible signature must propagate `TypeError`; an unavailable solver must retain the focused `MissingSynthesisSolverError`.

### OpenHENS Comparison Prerequisite

Run:

```bash
uv run pytest -q tests/packaging/test_openhens_comparison_prerequisite.py
```

A supported capability object must remain unmodified. An incomplete checkout must fail before creating an output directory and name the missing upstream capability.

### Documentation and Package Surface

Run the warning-as-error Sphinx build and isolated artifact smoke from the build instructions. Confirm that `docs/reference/api-lib.rst` and its navigation entry are absent and that the installed root package exports exactly two workflow entry points.

## Real Solver Integration

With Couenne and IPOPT available:

```bash
uv run pytest -q -m solver
```

Expected result for the implemented environment: 3 passed and 1 intentional nine-stream skip. The four-stream live regression requires at least 95 successful ESM branches while retaining exact accepted-design and cost checks.

No databases, web services, or cleanup steps are required. Temporary documentation, distribution, and virtual-environment directories may be discarded after inspection.
