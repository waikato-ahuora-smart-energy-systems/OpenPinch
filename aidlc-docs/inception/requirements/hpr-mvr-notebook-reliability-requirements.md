# HPR and MVR Notebook Reliability Requirements

Date: 2026-09-25. Status: approved.

## Intent Analysis

- **User request**: Ensure every notebook involving HPR or MVR targeting works
  properly, including correcting Notebook 10 so its CoolProp-backed examples
  demonstrate bounded convergence towards an optimal shared design across all
  operating periods while retaining useful evidence when an optional,
  deliberately harder configuration is infeasible.
- **Request type**: Brownfield tutorial reliability correction with regression
  coverage.
- **Scope**: Generated notebooks 08 through 11, their authoritative generator,
  compact notebook-local result presentation, and focused notebook, HPR, MVR,
  performance-map, and multiperiod tests.
- **Complexity**: Standard. The public targeting services already support the
  required targeting capabilities; the correction must select explicit
  topology and budget settings, distinguish mandatory and optional outcomes,
  and must not accidentally invoke Notebook 10's shared optimization once per
  period.
- **Authorization boundary**: No production API change is required unless
  implementation exposes a separate production defect that cannot be corrected
  within the tutorial and test surfaces.

## Notebook Inventory and Investigation Basis

Static inspection of every packaged notebook and its generator identified four
notebooks that invoke HPR or MVR targeting:

| Notebook | Targeting surface | Required healthy outcome |
|---|---|---|
| 08 - Carnot Heat Pump and Refrigeration | Carnot heat pump, Carnot refrigeration, residual utility placement | Both HPR targets solve under explicit bounds; downstream residual analysis remains valid. |
| 09 - Vapour Compression HPR Targets and Performance Maps | CoolProp VC heat pump and refrigeration, performance map, optional TESPy and unavailable Brayton screens | CoolProp heat pump and refrigeration solve under explicit bounds and the performance map is valid; optional backend/unavailable-method outcomes remain honest and typed. |
| 10 - Multiperiod Heat Pumps | Carnot, CoolProp VC, refrigeration, and VC+MVR multiperiod targeting | Five separately initiated shared-design optimizations solve across all three periods; an optional advanced cascade may report typed infeasibility. |
| 11 - Process MVR and VC Cascade | Direct process MVR plus optimized VC+MVR | Direct process MVR, serial/parallel period replay, and bounded optimized VC+MVR complete with valid records. |

No other packaged notebook invokes an HPR or MVR target. Notebook 06 performs
ordinary multiperiod heat integration and is therefore outside this targeting
scope.

The existing combined `slow-hpr` execution profile began notebooks 08 and 09
but remained inside Notebook 10's optimizer until manually interrupted after
243 seconds. Because the four notebooks execute inside one parametrized test,
that profile did not report an individual pass even for the notebooks that had
already completed.

Notebook 08 completed successfully in the baseline run. A bounded probe using
one restart, 20 iterations, and 50 evaluations also solved both Carnot targets
in under 0.1 seconds of targeting time.

Notebook 09 completed only because its helper converted optional and
refrigeration failures to generic status dictionaries. Its primary CoolProp
heat-pump target succeeded, but inherited large work limits and achieved only a
very small economic load. Its VC refrigeration call omitted explicit topology
and failed generically. With one condenser, one evaporator, one restart, 20
iterations, and 50 evaluations, the same CoolProp heat-pump and refrigeration
calls both solved in approximately 2.8 seconds combined. The bounded heat-pump
result achieved about 147.1 kW of its 187.5 kW selected ceiling and the bounded
refrigeration result achieved about 167.9 kW of its 250.0 kW selected ceiling.

Notebook 11 completed all code cells in approximately 3.3 seconds. Its direct
process-MVR stage results were populated, serial and parallel multiperiod heat
integration agreed, and its bounded optimized VC+MVR assertion passed.

### Notebook 10 failure basis

The checked-in executed notebook currently reports generic failures for
vapour-compression heat pumping, vapour-compression refrigeration, and VC+MVR.
Exact replay produced 1,174, 1,174, and 1,336 physically infeasible candidate
evaluations respectively. Those calls silently inherited a three-condenser,
two-evaporator cascade and search limits of 300 iterations and 1,000,000
evaluations.

An explicit one-condenser, one-evaporator, one-MVR-stage configuration with one
restart, 20 iterations, and 50 evaluations solved all three simulated methods
independently for `turndown`, `base`, and `peak`. A second probe enabled the
existing shared-design multiperiod optimization and solved all five tutorial
technologies across the same three periods in approximately 13 seconds:

- Carnot heat pumping;
- Carnot refrigeration;
- vapour-compression heat pumping with water;
- vapour-compression refrigeration with ammonia; and
- vapour compression with mechanical vapour recompression.

For shared optimization, a scalar targeting method is called once for a chosen
reporting period. The optimizer evaluates one design vector across all prepared
periods and returns the ordered period outputs, period weights, weighted result,
and shared design vector in `hpr_details`. Calling the same method through
`target.all_periods` would incorrectly repeat the shared optimization for each
selected period.

## Functional Requirements

### FR-1: Canonical generated notebook set

`scripts/generate_tutorial_notebooks.py` shall remain the authoritative source
for notebooks 08, 09, 10, and 11. Each generated notebook shall match its
generator token-for-token under the repository's drift check.

All four regenerated notebooks shall be source-only artifacts: every code cell
has a null execution count and no saved outputs. Notebook 10's currently saved
execution output shall not be retained in the canonical notebook or copied to a
new artifact.

### FR-2: Notebook-wide bounded targeting

Every optimizer-backed HPR or MVR call in notebooks 08 through 11 shall specify
explicit work limits supported by its public interface. Required CoolProp and
Carnot examples shall use one restart, at most 20 iterations, and at most 50
objective evaluations unless a smaller existing bound is proven sufficient.
The existing bounded VC+MVR example in Notebook 11 may retain its stricter
iteration limit, but its success shall be asserted.

Every required simulated VC or VC+MVR example shall explicitly state its stage
topology and working fluid. It shall not inherit the repository's more complex
three-condenser/two-evaporator defaults.

### FR-3: Notebook 08 required outcomes

Notebook 08 shall successfully solve its Carnot heat-pump and Carnot
refrigeration targets under explicit bounds. Its heat-pump residual utility
allocation, utility-placement optimization, derived residual problem, plots,
and summaries shall remain executable and shall refer to the intended target.

The notebook shall assert meaningful target success and expose compact load and
objective evidence rather than relying only on absence of an exception.

### FR-4: Notebook 09 required and optional outcomes

Notebook 09 shall successfully solve bounded one-condenser, one-evaporator
CoolProp heat-pump and refrigeration targets. The primary heat-pump result shall
retain a detached target simulation record and shall generate a valid plain-data
performance map from that record.

TESPy shall remain an explicit optional backend selection with no fallback to
CoolProp. Brayton methods may remain explicitly unavailable while solver repair
is pending. Optional dependency absence, declared method unavailability, and
typed HPR infeasibility shall be reported distinctly; they shall not all be
collapsed into the same generic `no feasible solution` status.

The notebook shall assert the required CoolProp outcomes. It shall not use a
catch-and-continue wrapper to make a required CoolProp failure look like a
successful notebook execution.

### FR-5: Notebook 10 five bounded shared-design optimizations

The main notebook workflow shall run one shared-design multiperiod optimization
for each of the five technologies listed in the investigation basis. Every call
shall explicitly enable multiperiod optimization and shall use a scalar public
targeting method exactly once, with a clearly identified reporting period such
as `base`.

The simulated vapour-compression and MVR examples shall explicitly request:

- one condenser;
- one evaporator;
- one MVR stage where MVR applies;
- one restart;
- at most 20 optimizer iterations; and
- at most 50 objective evaluations.

The Carnot calls shall also use explicit bounded arguments appropriate to their
public interfaces. Notebook behavior shall not depend on inherited complex
topology or large optimizer defaults.

### FR-6: Notebook 10 shared-period result proof

Each successful main-workflow result shall prove that a single returned design
was evaluated across `turndown`, `base`, and `peak`. The notebook shall inspect
the shared `hpr_details`, including the design vector, ordered period outputs,
period weights, and weighted result, rather than treating three independently
retuned designs as a shared installed design.

The presentation shall make clear that every technology is optimized
separately. It shall not imply that OpenPinch performs a meta-optimization to
select the best technology among the five.

### FR-7: Compact, accurate presentation

All four notebooks shall present compact results suitable for interactive and
automated execution. They shall include, as applicable:

- technology name and completion status;
- reporting period and all evaluated period identifiers;
- objective or weighted result;
- shared design vector or a bounded summary of it; and
- concise structured failure information.

It shall not emit the unbounded nested `TargetOutput` representation that
obscures the solve outcome.

### FR-8: Notebook 10 optional advanced cascade screen

After the five successful main examples, the notebook shall include one clearly
labelled optional advanced cascade feasibility screen. The screen shall use
explicit topology and work limits. It may either solve or return a typed
infeasibility outcome; it is not an acceptance oracle for the five main
examples.

The surrounding explanation shall distinguish an engineering infeasibility
result from a software execution failure and from the simpler shared-design
demonstration.

### FR-9: Structured failure contracts

Notebook-local screening helpers shall use the narrowest expected exception
boundary. Typed HPR infeasibility shall return plain, JSON-serializable data
containing:

- an accurate status that does not falsely describe independent replay as a
  failed shared design;
- the exception message; and
- bounded structured diagnostics derived from
  `HPRTargetingError.diagnostics`.

Optional dependency absence and explicit `NotImplementedError` shall use
separate statuses from thermodynamic infeasibility. Unexpected programming
errors shall continue to propagate. Helpers shall not fabricate success,
silently substitute a different technology, or discard the reason for failure.

### FR-10: Notebook 11 process-MVR and optimized VC+MVR proof

Notebook 11 shall prove that direct process MVR produces non-empty per-period
stage results and valid compressor work, and that serial and parallel
multiperiod heat-integration results agree. Its optimized VC+MVR target shall
complete under explicit bounds and expose non-empty detached loop records.

The notebook shall preserve component activation/deactivation behavior and
compactly present process-MVR and optimized-cascade evidence.

### FR-11: Reproducible generation and execution

Notebook generation shall be deterministic. Repeated generation from the same
source and inputs shall produce the same notebook source. Any randomized test
data shall use the project seed `20260715` where the test framework exposes a
seed.

## Non-Functional Requirements

### NFR-1: Runtime boundedness and attribution

Every optimizer-backed HPR or MVR call executed by notebooks 08 through 11
shall have explicit restart, iteration, and evaluation limits where supported.
The slow-HPR profile shall complete without relying on the inherited
million-evaluation limit.

Notebook execution tests shall expose each notebook as a distinct case so a
slow or failing notebook does not hide the completed status of the others.

### NFR-2: Pedagogical correctness

Notebook prose and output shall distinguish:

1. one design evaluated across several operating periods;
2. separately optimized per-period designs; and
3. a harder feasibility screen that may be physically infeasible.

It shall also distinguish required examples from optional backends and methods
that are explicitly unavailable.

### NFR-3: Public contract preservation

The correction shall use existing public targeting capabilities. It shall not
weaken typed HPR failures, optimizer hard caps, detached results, or the public
result contracts established by the prior HPR/MVR reliability work.

### NFR-4: Portable diagnostics

Failure details exposed by the notebook helper shall contain only detached
plain data suitable for JSON serialization. Diagnostic collections and strings
shall be bounded so an infeasible screen cannot flood notebook output.

## Acceptance Criteria

1. Generated notebooks 08, 09, 10, and 11 are source-only and pass the generator
   drift assertion.
2. Notebook 08's bounded Carnot heat pump and refrigeration both succeed and
   its downstream residual workflows execute.
3. Notebook 09's bounded CoolProp heat pump and refrigeration both succeed; the
   heat-pump record and generated performance map are valid.
4. Notebook 09 reports optional TESPy absence/failure and Brayton unavailability
   honestly and separately without masking required CoolProp failures.
5. Notebook 10's main workflow runs exactly five separately initiated
   shared-design multiperiod optimizations.
6. Each of Notebook 10's five main results succeeds with a shared design whose
   period outputs cover `turndown`, `base`, and `peak`.
7. Notebook 10's simulated VC, refrigeration, and VC+MVR calls use the explicit
   single-stage topology and verified hard work limits.
8. Notebook 11's direct process MVR and bounded optimized VC+MVR both succeed,
   stage and loop records are non-empty, and serial/parallel replay agrees.
9. All four notebooks display compact summaries and do not dump unbounded
   nested target representations.
10. Notebook 10's optional advanced cascade screen is labelled as such and
    either succeeds or returns an accurate typed failure summary with
    JSON-serializable bounded diagnostics.
11. Unexpected exceptions from screening helpers are not converted into
   ordinary infeasibility.
12. Focused per-notebook execution, generator drift, HPR/MVR regression, and
    applicable property-based tests pass.
13. Relevant broader tests confirm that existing HPR/MVR public behavior is not
   regressed.

## Verification Requirements

- Add or refine an exact generator-versus-notebook drift test.
- Preserve or add a source-only notebook invariant test.
- Parameterize slow-HPR execution by notebook and execute all four notebooks
  through complete code-cell execution oracles.
- Add focused assertions for Notebook 08's two bounded Carnot successes,
  Notebook 09's two required CoolProp successes and performance map, and
  Notebook 11's direct and optimized MVR evidence.
- Add a focused real-service regression proving that all five bounded shared
  designs contain outputs for all three expected periods.
- Test the advanced screening helper's successful result, typed failure, bounded
  diagnostic conversion, and propagation of unexpected errors.
- Run the relevant focused HPR/MVR and notebook suites, followed by the broadest
  practical regression suite defined during Build and Test.

## Property-Based Testing Extension

The Property-Based Testing extension is enabled in full. The implementation and
test design shall satisfy the applicable rules as follows:

| Rule | Applicability and required evidence |
|---|---|
| PBT-01 | Define properties for bounded plain-data diagnostics, status vocabulary, and period coverage independently of individual examples. |
| PBT-02 | Exercise JSON round trips for generated valid diagnostic summaries when a new conversion helper is introduced. |
| PBT-03 | Vary valid period identifiers, diagnostic counts, messages, and bounded collections while preserving contract invariants. |
| PBT-04 | Apply idempotence or metamorphic checks to pure normalization only if such normalization is introduced; otherwise N/A. |
| PBT-05 | Use existing public contract objects and explicit per-notebook example oracles to confirm compact summaries agree with source results. |
| PBT-06 | N/A unless implementation introduces a stateful helper; the planned notebook conversion is pure and local. |
| PBT-07 | Constrain strategies to valid diagnostic shapes and bounded collection sizes. |
| PBT-08 | Use reproducible seed `20260715` where supported and retain minimized failing examples. |
| PBT-09 | Use the repository's existing Hypothesis stack and settings rather than adding a second framework. |
| PBT-10 | Retain conventional example-based tests for every required real bounded solve and use properties as complementary contract coverage. |

Security Baseline and Resiliency Baseline are disabled for this local tutorial
correction and are not blocking constraints.

## Scope Boundaries

### In scope

- Generator and generated source-only notebooks 08, 09, 10, and 11.
- Notebook-local compact result and diagnostic helpers across that set.
- Focused per-notebook regression, execution, drift, and property-based tests.
- Minimal documentation needed to explain the shared-design semantics.

### Out of scope

- A new HPR optimization algorithm or public API.
- A meta-optimizer that selects among technologies.
- Production deployment, infrastructure, monitoring, or operational changes.
- Retaining executed notebook outputs in version control.
- Suppressing genuine infeasibility or treating every candidate set as required
  to solve.

## Extension Compliance

- **Property-Based Testing**: Compliant at requirements stage. Applicable
  properties, generators, reproducibility, framework, and complementary example
  tests are specified; conditional rules are marked N/A with rationale.
- **Security Baseline**: N/A because the user selected disabled.
- **Resiliency Baseline**: N/A because the user selected disabled.
