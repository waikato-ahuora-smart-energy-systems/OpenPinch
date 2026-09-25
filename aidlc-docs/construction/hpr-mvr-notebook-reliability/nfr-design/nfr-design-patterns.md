# HPR and MVR Notebook Reliability NFR Design Patterns

## Design Principles

1. The generator is the only authoring source; notebooks are regenerated
   artifacts.
2. Required tutorial outcomes execute directly and fail loudly.
3. Optional demonstrations use narrow typed outcome normalization.
4. Every optimizer call declares its topology, fluids, and hard work limits.
5. Notebook 10 evaluates each technology in an isolated problem instance and
   starts exactly one shared optimization for it.
6. Review output contains detached plain evidence rather than complete target
   objects.
7. Tests first prove source contracts, then execute each real notebook, then
   verify integration across the full profile.

## Pattern 1: Budget as Explicit Tutorial Code

Each notebook defines or spells out the public budget values at the call site:

```python
maximum_restarts=1,
maximum_iterations=20,
maximum_evaluations=50,
```

Notebook 11 may retain `maximum_iterations=3`, but its evaluation limit is
reduced to 50. Optional advanced screens may use smaller values. The explicit
arguments make budget behavior inspectable through static AST/source tests and
prevent stored configuration or default changes from expanding tutorial work.

This implements NFR-PERF-001, NFR-PERF-003, NFR-DET-001, and NFR-MAINT-001.

## Pattern 2: Explicit Single-Stage Configuration

Required simulated calls provide `condensers=1` and `evaporators=1`. VC+MVR
also provides `mvr_stages=1`, explicit VC refrigerants, and explicit MVR fluids.
Notebook 09 uses water for heat pumping and ammonia for refrigeration. Notebook
10 uses the same reviewed fluids for those modes. Notebook 11 explicitly uses
water for both VC and MVR portions.

This implements NFR-PERF-002 and removes reliance on the three-condenser,
two-evaporator inherited topology that caused the observed failures.

## Pattern 3: Fresh-Problem Shared-Design Isolation

Notebook 10 uses a small source factory that creates a new
`PinchProblem("crude_preheat_train_multiperiod.json", ...)` for every
technology. Each technology then calls one scalar target method with:

- `period_id="base"` as the reporting period;
- `options={"HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True, ...}`;
- explicit load, topology, fluid, and budget arguments; and
- no `target.all_periods` wrapper.

Fresh problem instances ensure that the five alternatives are evaluated from
the same original process state rather than inheriting prior target mutations.
The selected-period result is only the reporting shell: its `hpr_details`
contains the one shared vector, all three period outputs, aligned weights, and
weighted result.

This implements NFR-PERF-004 and NFR-REL-002.

## Pattern 4: Required Success Versus Optional Outcome

Required calls are direct assignments followed by contract assertions. They are
not passed through a screening wrapper.

Optional calls use a notebook-local adapter with four plain statuses:

- `feasible`;
- `typed infeasible`;
- `optional dependency unavailable`; and
- `method unavailable`.

The adapter catches `HPRTargetingError`, `ImportError`, and
`NotImplementedError` in separate branches. It never catches `Exception`, never
changes backend or technology, and never retries. Notebook 09 uses this only for
the optional TESPy and declared-unavailable Brayton demonstrations. Notebook 10
uses an HPR-only variant for the optional advanced cascade screen.

This implements NFR-REL-001, NFR-REL-004, and NFR-REL-005.

## Pattern 5: Detached Plain-Data Presentation

A notebook-local pure summary function extracts only stable evidence from a
successful target:

- label and `feasible` status;
- backend and cycle/topology identity;
- selected and achieved load;
- finite objective or weighted result;
- period identifiers and weights when present;
- finite design-vector values when present; and
- stage/loop counts where applicable.

It does not retain the target, stream collections, NumPy arrays, plots, or
engine objects. Array-like values are converted to bounded Python lists.

A typed-failure helper returns the exception message plus
`error.diagnostics.model_dump(mode="json")`. The public schema already bounds
representative failures and strings; no raw candidate log is appended.

This implements NFR-DIAG-001 through NFR-DIAG-003 and NFR-MAINT-001.

## Pattern 6: Generator-First Regeneration

All content changes are made in `scripts/generate_tutorial_notebooks.py` first.
The generator then rewrites notebooks 08 through 11 with null execution counts
and empty outputs. Two isolated generation passes must be byte-identical, and
the checked-in notebooks must equal the generated form.

Notebook 10's current saved execution output is therefore replaced by the
approved canonical source-only form only during Code Generation.

This implements NFR-DET-001 and NFR-DET-002.

## Pattern 7: Attributable Layered Verification

The optional-profile harness factors one cell-execution helper and parametrizes
every slow-HPR notebook name as a separate pytest case. Solver and interactive
profiles remain independently gated by their existing environment selection.

Verification proceeds in layers:

1. source/AST checks prove explicit bounds, topology, scalar-versus-all-period
   call shape, and narrow exception handlers;
2. generation tests prove idempotence, source-only cells, and checked-in drift;
3. example tests prove notebook-specific result contracts;
4. property tests exercise plain summaries and diagnostic JSON across bounded
   valid generated data;
5. each real notebook executes separately;
6. the full slow-HPR profile and related HPR/MVR suites execute; and
7. packaging, documentation, lint, and broad regression gates execute.

This implements NFR-TEST-001, NFR-TEST-002, and NFR-DET-003.

## Pattern 8: Documentation and Manifest Synchronization

The tutorial coverage generator/CSV, notebook-series description, and HPR guide
are updated with the actual generated behavior. Notebook 10 documentation
describes the scalar call that enables shared-design optimization and contrasts
it with independent `target.all_periods` replay. Existing all-period API tests
remain the execution evidence for independent replay.

This implements NFR-MAINT-002.

## Property-Based Testing Design

- Extract notebook-local pure function definitions from generated source with
  `ast` for isolated testing; do not execute a thermodynamic solve for every
  Hypothesis example.
- Generate contract-valid `HPRFailureSummary` values and small valid result
  doubles within public field bounds.
- Assert JSON round-trip preservation, bounded representative failures,
  idempotent plain normalization when present, stable status vocabulary,
  finite scalar/list output, and period-set preservation.
- Keep real notebook execution as a separate example oracle; property tests do
  not replace CoolProp/MVR integration tests.
- Use existing Hypothesis settings and seed `20260715` where supported.

## Resilience, Scalability, Security, and Infrastructure

- **Resilience**: fail-fast required calls and typed optional outcomes; no retry,
  fallback, checkpoint, or circuit breaker.
- **Scalability**: fixed bounded local work; no horizontal or distributed
  component.
- **Security**: disabled extension; no new security boundary.
- **Infrastructure**: none.
- **Operations**: none.

## Extension Compliance

- Property-Based Testing: compliant; pure boundaries and integration oracles
  are separated and all applicable PBT requirements have a design mapping.
- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.
