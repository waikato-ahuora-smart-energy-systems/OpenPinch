# HPR and MVR Notebook Reliability Logical Components

## Component 1: Tutorial Generator

**Owner**: `scripts/generate_tutorial_notebooks.py`

Responsibilities:

- author notebook code, prose, review cells, and metadata;
- reuse generator-level source fragments for compact summaries and optional
  outcome handling where doing so prevents four copies from drifting;
- emit deterministic notebook v4 JSON with no execution output; and
- retain public-only notebook imports.

No targeting logic moves into the generator. It composes existing public calls.

## Component 2: Notebook 08 Carnot Workflow

Required calls:

| Call | Topology | Budget | Result evidence |
|---|---|---|---|
| Carnot heat pump | 1 condenser, 1 evaporator | 1 restart, 20 iterations, 50 evaluations | successful load/objective plus downstream residual workflows |
| Carnot refrigeration | 1 condenser, 1 evaporator | 1 restart, 20 iterations, 50 evaluations | successful load/objective and plots |

The existing residual utility placement and problem derivation continue to use
the successful heat-pump target. The review cell displays summaries and figures,
not the entire residual target object.

## Component 3: Notebook 09 Simulated HPR Workflow

Required calls:

| Call | Fluid/topology | Budget | Result evidence |
|---|---|---|---|
| CoolProp heat pump | water, 1 by 1 | 1 restart, 20 iterations, 50 evaluations | successful detached record and valid performance map |
| CoolProp refrigeration | ammonia, 1 by 1 | 1 restart, 20 iterations, 50 evaluations | successful compact target summary |

Optional calls:

- explicit TESPy mixture heat pump with the same work caps; and
- Brayton heat-pump and refrigeration methods with explicit work caps.

The optional adapter distinguishes dependency absence, method unavailability,
and typed infeasibility. The performance-map path depends only on the required
CoolProp heat-pump result and cannot silently consume an optional fallback.

## Component 4: Notebook 10 Shared-Design Workflow

Notebook 10 creates five fresh problem instances and starts these scalar calls:

| Technology | Fluid/topology | Shared setting | Budget |
|---|---|---|---|
| Carnot heat pump | 1 by 1 | enabled, reporting period `base` | 1 restart, 20 iterations, 50 evaluations |
| Carnot refrigeration | 1 by 1 | enabled, reporting period `base` | 1 restart, 20 iterations, 50 evaluations |
| CoolProp VC heat pump | water, 1 by 1 | enabled, reporting period `base` | 1 restart, 20 iterations, 50 evaluations |
| CoolProp VC refrigeration | ammonia, 1 by 1 | enabled, reporting period `base` | 1 restart, 20 iterations, 50 evaluations |
| CoolProp VC+MVR | water VC and MVR fluids, 1 by 1 plus 1 MVR stage | enabled, reporting period `base` | 1 restart, 20 iterations, 50 evaluations |

Every result must contain the same period set: `turndown`, `base`, and `peak`.
The comparison is a display of five separately optimized alternatives, not a
joint technology selector.

The optional advanced screen uses a fresh sixth problem, explicitly requests a
multi-stage cascade, enables the same shared-design mode, applies smaller hard
limits, and catches only `HPRTargetingError`. Either a compact feasible summary
or bounded typed diagnostics is valid.

## Component 5: Notebook 11 Process-MVR Workflow

The direct process-MVR component remains deterministic and non-optimizer-backed.
It must retain populated stage results, finite compressor work, component
activation/deactivation, and equivalent serial/parallel heat-integration
results.

The separate optimized VC+MVR example uses water for both VC and MVR fluids,
one condenser, one evaporator, one MVR stage, one restart, no more than 20
iterations, and no more than 50 evaluations. It executes directly, asserts
success, and proves non-empty detached loop records.

## Component 6: Notebook-Local Plain Helpers

Logical functions:

- `summarize_hpr_target(label, target)` returns bounded plain success evidence;
- `summarize_hpr_failure(error)` returns message plus JSON-mode diagnostics;
  and
- `screen_optional_hpr(method, **arguments)` provides the four optional status
  outcomes only where the notebook explicitly declares an optional call.

The functions remain self-contained notebook code, optionally authored from a
shared generator source fragment. They are not added to the OpenPinch public or
internal runtime packages.

## Component 7: Per-Notebook Execution Harness

**Owner**: `tests/packaging/test_notebooks.py`, with a focused companion test
module only if needed for readability.

Responsibilities:

- resolve slow-HPR notebook names from the canonical manifest;
- emit one pytest parameter case per notebook;
- execute all code cells in order in an isolated temporary working directory;
- preserve the existing opt-in environment/profile contract;
- report notebook name and cell index on failure; and
- keep interactive UI stubbing limited to the interactive profile.

The harness does not swallow notebook exceptions or impose a different
optimization configuration.

## Component 8: Static and Property Contract Tests

Static tests inspect generated source/AST for:

- explicit limits and topology;
- no required call inside optional wrappers;
- one scalar shared call per Notebook 10 technology;
- no shared call through `target.all_periods`;
- narrow exception handlers;
- compact review variables; and
- source-only generator synchronization.

Property tests isolate pure helper function definitions and exercise valid
bounded diagnostics and result doubles without invoking CoolProp per example.
Real example tests remain responsible for thermodynamic integration.

## Component 9: Tutorial Metadata and Guidance

**Owners**:

- `scripts/generate_tutorial_coverage.py` and
  `docs/_data/tutorial-coverage.csv`;
- `docs/examples/notebook-series.rst`; and
- `docs/guides/heat-pump-workflows.rst`.

These files describe Notebook 10 as a shared installed-design study, preserve
independent `target.all_periods` replay as a distinct public capability, and
identify the real execution evidence accurately.

## Component 10: Existing Runtime Providers

The following are dependencies, not planned change owners:

- public target and component accessors;
- HPR/MVR analysis and optimizer adapters;
- multiperiod preparation/evaluation/aggregation;
- HPR target, simulation-record, search-budget, and failure-summary contracts;
  and
- performance-map generation.

If a required notebook cannot pass through these existing contracts under the
verified settings, implementation stops and reports the runtime defect before
changing this boundary.

## Traceability Matrix

| NFR group | Design components |
|---|---|
| Performance and bounds | 1, 2, 3, 4, 5, 8 |
| Reliability and correctness | 2, 3, 4, 5, 6, 7 |
| Diagnostics and compact output | 1, 3, 4, 6, 8 |
| Determinism | 1, 7, 8 |
| Test attribution and maintainability | 1, 7, 8, 9 |
| Scalability/availability/security/operations | no new component; explicit N/A or bounded local execution |
| Property-Based Testing | 6 and 8, complemented by real components 2 through 5 |

## Extension Compliance

- Property-Based Testing: compliant through Components 6 and 8 plus real
  integration Components 2 through 5.
- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.
