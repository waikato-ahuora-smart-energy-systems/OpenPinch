# Package Architecture Modernization Design

## Design Intent

The design optimizes for change isolation: one requirement should normally
touch its owner, its explicit callers, and behavioural tests rather than a
cross-package chain. The only current compatibility boundary is
`OpenPinch.main.pinch_analysis_service`; all deeper paths are implementation
details even when documented for contributors.

## Owner Model

| Owner | Responsibility | Inward dependencies |
|---|---|---|
| `domain` | Business state, invariants, arithmetic, identity, indexing, parent-owned records | domain peers and numerical/value libraries |
| `contracts` | Request, response, report, HPR, HEN, and workspace wire structures | contracts and domain |
| `optimisation` | Reusable scalar problems, methods, candidate ordering, execution, and backends | optimisation peers, NumPy, SciPy |
| `adapters` | JSON, CSV, workbook, filesystem, packaged-source, and optional-dependency translation | adapters, contracts, domain |
| `analysis` | Deterministic thermal, HPR, HEN, graph-data, exergy, power, and engineering calculations | analysis, contracts, domain, optimisation, narrow adapter leaves |
| `application` | Single-case and workspace orchestration, replay, caching, period execution, and comparison | inward owners and lazy presentation boundaries |
| `presentation` | Tables, reports, workbooks, Plotly conversion, dashboards, and network-grid rendering | presentation, analysis, adapters, contracts, domain |

## Dependency Rules

- Dependencies point inward toward domain and contracts.
- Domain never imports application, analysis, adapters, optimisation, or
  presentation.
- Contracts never import application, analysis, adapters, optimisation, or
  presentation.
- Optimisation has no HPR dependency and no mutable backend registry.
- Application does not import concrete solver backends, Plotly, Streamlit, or
  filesystem implementation classes.
- Presentation and infrastructure dependencies are loaded by their owning
  leaves so core imports do not require optional packages.
- A small exact set of existing HEN and typing edges is enumerated by the AST
  boundary test. New files cannot inherit those exceptions implicitly.

## Composition Decisions

- `PinchProblem` remains a composition hub for one use case. Independent
  loading, validation, targeting, replay, period aggregation, result shaping,
  reporting, graphing, and dashboard concerns delegate to concrete owners.
- `PinchWorkspace` owns named-case coordination and delegates state, execution,
  comparison, and view shaping to private application helpers.
- HEN model classes remain equation-model coordinators. Parameter loading,
  equations, approach constraints, execution, warm starts, objectives,
  post-processing, verification, and extraction receive explicit model state.
- Heat-pump analysis translates HPR objective semantics into the reusable
  optimisation model at one adapter and translates ranked candidates back.
- Runtime records are constructed by and observed through their parent. No
  public record factory, alias, or pickle migration is retained.

## Simplicity Decisions

- Package `__init__.py` files are marker modules, not API barrels.
- Concrete modules may declare `__all__` for local clarity, but may not re-export
  objects owned by a different module as a compatibility surface.
- Small modules are retained only when they own a real descriptor, error,
  type, equation variant, optional-dependency boundary, or independently
  changing algorithm.
- No line-count target, mixin hierarchy, service locator, plugin registry, or
  protocol is added without a current second implementation need.

## Test Design

- `tests/e2e/test_main.py` protects the only external contract using caller
  mappings and caller-visible outcomes.
- Tests mirror application, domain, analysis, optimisation, adapters,
  presentation, and contracts. Architecture and packaging gates remain
  separate.
- Exact numerical fixtures protect HPR and HEN algorithms. Property tests cover
  round trips, domain transactions, ordering, conservation, and generated
  optimisation candidates with seed `20260715` and normal shrinking.
- Fresh-process and wheel-installed tests prove optional-dependency isolation,
  retired-path absence, and source-checkout independence.

## As-Built Conformance Review

- `OpenPinch/__init__.py` and owner `__init__.py` files are import-free markers.
  `OpenPinch/main.py` remains a minimal ten-statement boundary and retains
  `__all__ = ["pinch_analysis_service"]`.
- Core classes have concrete owners: values, streams, collections, problem
  tables, zones, exchangers, and exchanger networks are in `domain`; problem
  and workspace use cases are in `application`; wire models are in
  `contracts`.
- Shared optimisation has no domain, HPR, application, adapter, or presentation
  dependency. Generic scalar tests and the HPR adapter demonstrate both sides
  of the intended reuse boundary.
- `PinchProblem` remains the use-case composition hub. Loading, semantic
  validation, targeting dispatch/execution, periods, result extraction,
  reporting, graph requests, and dashboard requests have independent owners;
  lifecycle, use-case identity, and cache coordination remain on the parent.
- The over-splitting review removed forwarding modules, imported-name exports,
  ignored compatibility parameters, and modules with no independent owner.
  Small remaining modules own real errors, descriptors, immutable state,
  optional-dependency boundaries, or equation variants.
- The dependency test records a small exact set of existing HEN/application
  and presentation/application edges. New files cannot use those exceptions,
  so the residual debt is visible and cannot spread silently.
- No compatibility facade, migration path, service locator, mixin hierarchy,
  mutable registry, or speculative algebraic-optimisation layer was added.

The final distribution has 326 entries and 23,785 executable statements are
measured by coverage. These are observations, not module-size targets.
