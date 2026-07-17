# Package Architecture Modernization Implementation Summary

## Outcome

OpenPinch now has owner-oriented package boundaries designed around reasons to
change. The only compatibility-protected Python entry point is:

```python
from OpenPinch.main import pinch_analysis_service
```

`OpenPinch/main.py` remains minimal. Its function signature, validation order,
exceptions, output model, field ordering, serialization, and representative
numerical results are unchanged. The root package is an import-free marker.

## As-Built Owner Structure

| Package | Ownership |
|---|---|
| `OpenPinch.domain` | Business state, invariants, arithmetic, indexing, and parent-owned runtime records |
| `OpenPinch.contracts` | Main request/response and other wire-format models |
| `OpenPinch.optimisation` | Reusable scalar optimisation models, candidates, execution, and numerical backends |
| `OpenPinch.application` | Problem/workspace use cases, state, replay, caching, periods, and coordination |
| `OpenPinch.analysis` | Thermal, targeting, heat-pump, HEN, graph-data, exergy, power, and engineering algorithms |
| `OpenPinch.adapters` | JSON, CSV, workbook, filesystem, resources, and optional-dependency translation |
| `OpenPinch.presentation` | Reports, tables, workbooks, Plotly conversion, dashboard, and network-grid rendering |

The core classes now have these concrete owners:

| Core class | Concrete owner |
|---|---|
| `Value` | `OpenPinch.domain.value` |
| `Stream` | `OpenPinch.domain.stream` |
| `StreamCollection` | `OpenPinch.domain.stream_collection` |
| `ProblemTable` | `OpenPinch.domain.problem_table` |
| `Zone` | `OpenPinch.domain.zone` |
| `HeatExchanger` | `OpenPinch.domain.heat_exchanger` |
| `HeatExchangerNetwork` | `OpenPinch.domain.heat_exchanger_network` |
| `PinchProblem` | `OpenPinch.application.problem` |
| `PinchWorkspace` | `OpenPinch.application.workspace` |

`StreamSegment`, `HeatExchangerPeriodState`, and
`HeatExchangerAreaSlice` retain their diagnostic model names but are private to
their parent packages. Process MVR, multiperiod HPR, graph, dashboard, grid,
evolution, and solver records are likewise owned by the service that creates
them. Parents normalize caller mappings and schemas and remain responsible for
mutation and rollback.

## Major Changes

- Split stream transactions, value coercion/units, collection projections, and
  problem-table interval insertion into private domain owners while retaining
  aggregate state and public behaviour on the parent classes.
- Moved input/output and synthesis schemas into concrete contract modules with
  no reverse barrel imports.
- Decomposed problem and workspace orchestration by loading, semantics,
  validation, targeting, periods, results, state, comparison, and views.
- Separated deterministic engineering calculations from I/O and presentation.
  Optional Plotly, Streamlit, workbook, and solver dependencies are loaded at
  their owning leaves.
- Introduced a package-level optimisation service that accepts generic scalar
  problems. Heat-pump analysis crosses one adapter that owns HPR-specific
  objective, penalty, cost, and result translation.
- Decomposed HEN base, stagewise, pinch-design, solver-extraction, and reporting
  internals with explicit model state while preserving axes, equation order,
  warm starts, tolerances, period ordering, and result structures.
- Removed `OpenPinch.classes`, `OpenPinch.lib`, `OpenPinch.services`,
  `OpenPinch.utils`, `OpenPinch.streamlit_webviewer`, compatibility barrels,
  forwarding aliases, pickle shims, ignored compatibility arguments, and
  public-looking internal records.
- Reorganized tests by observable owner and added AST dependency, cold-import,
  artifact, entrypoint, and no-facade gates. The external suite uses caller
  mappings and caller-visible outcomes.

## Composition and Simplicity Review

`PinchProblem` remains intentionally responsible for one use-case lifecycle,
identity, and cache coordination. Concerns with independent reasons to change
now delegate to concrete owners. Moving the remaining core lifecycle would
create pass-through machinery without reducing change coupling.

The package-by-package over-splitting review removed thin forwarding modules
and imported-name exports. Small files remain only where they establish a real
boundary: an error or descriptor, immutable state, optional dependency,
equation variant, serialization format, or independently changing algorithm.
No line-count target, mixin, service locator, mutable registry, or speculative
protocol was introduced.

## Post-Review Source-Tracking Correction

The HEN `results` owner was initially present in the working tree and release
artifacts but hidden from Git by the repository-wide `results/` ignore rule.
That made local imports pass while a clean checkout could omit `assembly.py`,
`selection.py`, `seeds.py`, and the package marker. The ignore policy now has a
narrow exception for this Python source package.

Two regressions close the gap: every Python source under `OpenPinch` must be
visible to Git even when checked with `--no-index`, and wheel/sdist inspection
requires the HEN result assembly owner. This protects clean-checkout Sphinx and
packaging builds rather than only working-tree builds.

The missing package also explained the reported Ruff failure. A Git-index
snapshot without `results` reproduced five `I001` import-order diagnostics:
Ruff could not classify those HEN imports as first-party. Adding the four source
modules and corrected ignore policy made the unchanged snapshot pass. No
import-order workaround or per-file suppression was required.

The same ignore-policy audit found the approved architecture checklist under an
ignored `plans/` directory. A specific documentation exception now keeps that
checklist visible without exposing generated solver plan directories.

## Quality Review

| Quality question | Score | Evidence and deduction |
|---|---:|---|
| Ease of Change | 9.4/10 | A new service can import generic optimisation models/service without heat-pump code; owner tests localize changes. Deducted for a small explicit set of historical HEN/application edges. |
| Simplicity | 9.1/10 | Marker packages, concrete owners, explicit composition, and removal of facades/pass-throughs reduce machinery. Deducted because intrinsically complex solver kernels and several large domain aggregates remain. |
| Behavioural Tests | 9.5/10 | The 59-case main suite, exact HPR/HEN fixtures, domain outcomes, and generated invariants dominate. Deducted because a narrow set of equation-order tests must remain structural. |
| Clear Boundaries | 9.2/10 | AST dependency, owner, cold-import, and optional-dependency tests pass. Deducted for the enumerated HEN/application and presentation/application exceptions retained to avoid speculative redesign. |
| Low Coupling | 9.2/10 | Inputs and model state are explicit, optional libraries are isolated, and no mutable global registry exists. Deducted for unavoidable coupling inside equation-model composition. |
| Project Coherence | 9.3/10 | Package, function, test, error, and private-owner naming follow one model, and docs match the tree. Deducted for historical terminology retained where changing it would obscure established engineering concepts. |

**Overall quality score: 9.3/10.** This is the arithmetic mean rounded to one
decimal place and exceeds the 8.8 gate.

## Extension Compliance

- PBT-02: compliant through input/output and value serialization round trips.
- PBT-03: compliant through transaction, ordering, conservation, interval, and
  candidate invariants.
- PBT-07: compliant through centralized realistic domain and constrained
  optimisation strategies.
- PBT-08: compliant; shrinking remains enabled and seed `20260715` is recorded.
- PBT-09: compliant; Hypothesis remains the selected Python framework.
- Security Baseline: N/A because disabled.
- Resiliency Baseline: N/A because disabled.
