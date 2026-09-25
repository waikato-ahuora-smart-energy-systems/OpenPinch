# HPR and MVR Benchmark E2E Code Generation Summary

## Outcome

The standard 54-problem targeting corpus now drives one deterministic six-
profile real-CoolProp HPR matrix in the ordinary non-solver test tier. Every
case must solve, return a documented no-op, or raise bounded
`HPRTargetingError`; unexpected exceptions propagate. Six named sentinels must
also demonstrate material progress toward and final selection of the best
viable objective observed within the bounded real search.

A separate packaged `process_mvr.json` workflow exercises two-stage direct
process MVR and downstream targeting through public accessors.

## Created files

- `tests/e2e/cases.py`: single immutable sorted owner for standard problem
  discovery.
- `tests/e2e/hpr_benchmark.py`: frozen profiles, stable assignments, state and
  outcome contracts, typed-failure validation, real search observation and pure
  convergence analysis.
- `tests/e2e/test_hpr_benchmark_helpers.py`: fixed examples and Hypothesis
  properties for discovery, assignment, classification, atomicity, diagnostics,
  observer delegation and convergence.
- `tests/e2e/test_hpr_mvr.py`: complete 54-case HPR matrix, six sentinels and
  direct process-MVR workflow.

## Modified files

- `tests/e2e/test_main.py`: imports the shared corpus owner rather than defining
  a second discovery function.
- `OpenPinch/analysis/heat_pumps/service.py`: aligns ambient cascade columns to
  the shared temperature grid while preserving duplicate target-temperature
  rows.
- `OpenPinch/analysis/heat_pumps/common/postprocessing.py`: collapses equivalent
  duplicate temperatures only at the strict detached residual-profile contract
  edge and rejects conflicting duplicates.
- `tests/analysis/heat_pumps/test_targeting.py`: pins ambient alignment across
  duplicate target temperatures.
- `tests/analysis/heat_pumps/test_hpr_residual_regressions.py`: pins equivalent
  duplicate collapse and conflicting-value rejection.

No dependency, lockfile, marker, CI workflow, public API, database, frontend,
infrastructure or deployment artifact changed.

## Matrix and budgets

| Profile | Cases | Selected-cycle allowance | Total observed public-search bound | Sentinel |
|---|---:|---:|---:|---|
| Direct cascade VC heat pump | 9 | 12 | 24 | `p_Adjiman et al.json` |
| Utility parallel VC heat pump | 9 | 12 | 24 | `p_Ahmad (example 1).json` |
| Direct cascade VC refrigeration | 9 | 12 | 24 | `p_Ahmad (example 2).json` |
| Utility parallel VC refrigeration | 9 | 12 | 24 | `p_Ahmad (example 3).json` |
| Direct optimized VC+MVR heat pump | 9 | 20 | 40 | `p_Pavao et al (example 1).json` |
| Utility optimized VC+MVR heat pump | 9 | 16 | 32 | `p_Feng et al (case study 1).json` |

The 12-evaluation allowance was sufficient for all four VC sentinels. Direct
MVR did not pass at 12, 16, 17, 18 or 19 and passed at 20. The selected utility
MVR sentinel did not pass from 12 through 15 and passed at 16. These are the
smallest stable values found by the explicit sequential probes and keep every
public call below the approved ceiling of 50 search observations.

The initial matrix and calibrated MVR passes classified the final cases as 40
solved, five legitimate no-ops and nine bounded typed failures. Each profile
contains nine cases. Counts describe the tested environment and are not a
universal feasibility guarantee.

## Convergence evidence

The observer wraps `evaluate_hpr_candidate`, delegates exactly once with
unchanged arguments, returns the real result, and records only search mode. The
profile filter selects the declared VC or VC+MVR objective and excludes the
separate Carnot initializer search.

For every sentinel, the independent analyzer proves:

- at least two distinct viable selected-cycle points;
- improvement beyond `1e-8 * max(1, abs(first viable objective))`;
- a non-increasing incumbent-best sequence;
- zero selected-to-best-observed gap within tolerance; and
- selected and total evaluation counts within the declared bounds.

Observed material improvements were approximately 12.22 million, 8.22
thousand, 1.997 billion, 11.02 million, 60.996 million and 86.601 billion in
profile order. This is bounded progress toward the best observed candidate. It
does not prove a global optimum.

## Production defects corrected

### Duplicate-temperature ambient alignment

`p_Ciric and Floudas.json` under utility parallel refrigeration and
`p_Fodor et al.json` under utility parallel heat pumping both raised raw NumPy
broadcast errors. Their utility problem tables contain a duplicate exact
temperature while the ambient cascade has a unique grid. Shared interval
insertion therefore produced 31 versus 30 rows.

The local alignment helper now maps each target temperature to the shared source
row within the domain tolerance and repeats the ambient value for duplicate
target rows. Missing shared coordinates still fail explicitly.

### Strict residual-profile contract

After fixing alignment, the same valid cases reached the detached
`HPRResidualProfile` boundary with equivalent duplicate rows, which correctly
requires strictly decreasing temperatures. Equivalent duplicates are now
collapsed only for this detached record. Conflicting net/heating/cooling values
at a duplicate temperature raise a clear `ValueError` rather than losing a
physical discontinuity.

Both original benchmark cases now solve. With their public indirect-integration
prerequisites captured before the HPR transaction, each adds exactly one HPR
target.

## Direct process-MVR proof

The packaged scenario creates a two-stage component from `Evaporator vapour`
with explicit temperature lift and efficiencies. It verifies registration,
activation, pressure increase, finite positive stage duty/work, replacement
streams, copied detached stage arrays and the absence of retained engine/model/
state fields. Downstream direct heat integration produces finite Qh/Qc/Qr and
serializable result and problem representations.

Existing invalid-input, dry/reduced fallback and second-stage atomicity cases
remain complementary coverage.

## Verification evidence

- Baseline direct-target and focused reliability gate: 92 passed.
- Shared-corpus and existing direct-target gate: 60 passed.
- Fixed-seed helper/property gate: 11 passed.
- Complete helper plus 54-case HPR matrix: 65 passed in 45.43 seconds.
- Direct process-MVR plus complementary regressions: 60 passed.
- Duplicate-temperature focused regressions: three passed.
- Affected HPR/MVR analysis, contracts, application and e2e non-solver gate:
  1,164 passed in 206.40 seconds.
- Architecture, cold-import and package-usability gate: 41 passed.
- Normal `not solver` collection: 55 tests in `test_hpr_mvr.py`.
- Ruff lint and formatting: passed for every affected Python file.
- Python compilation: passed for affected HPR and e2e packages.
- Patch whitespace and duplicate-file inspection: passed.

These counts overlap and must not be summed.

## Requirement traceability

| Requirements | Implementation and evidence |
|---|---|
| FR-01 through FR-03; VR-01/VR-02 | Shared discovery, immutable six-profile assignment, helper examples/properties and unchanged direct-target e2e. |
| FR-04/FR-05; NFR-01/NFR-02/NFR-06 | Real public CoolProp calls, one restart, explicit deterministic inputs and bounded profile-specific search controls. |
| FR-06 through FR-11; VR-03 through VR-05 | Three-state classifier, transaction snapshots, strict solved/failure contracts, case/profile IDs and full matrix. |
| FR-08A/FR-08B; VR-03A/VR-03B | Six explicit sentinel IDs, observation-only traces and independent convergence witness. |
| FR-12 through FR-14; VR-06 | Packaged direct process-MVR public workflow and retained focused regressions. |
| FR-15 through FR-17; VR-08 | Full exploration, two reproduced failure paths, owner-level regressions and production corrections. |
| VR-07 | Seeded affected gate, Ruff, formatting, compilation, architecture, collection and patch hygiene. |
| NFR-03 through NFR-05/NFR-07 | Fresh problems, pre/post snapshots, bounded diagnostics, shared test ownership and public invariants. |

## Property-Based Testing compliance

- **PBT-01**: assignment, outcome, state, convergence, cache, budget and
  serialization properties have explicit owners.
- **PBT-02/PBT-03**: constrained finite objective sequences and corpus sizes
  cover valid/boundary partitions against independent prefix-minimum and
  tolerance oracles.
- **PBT-04**: exact-point deduplication has repeated-point examples; no new
  idempotent production operation is claimed.
- **PBT-05**: the existing standard corpus is the sole discovery oracle.
- **PBT-06**: fresh e2e state and existing stateful HPR cache properties cover
  mutable transitions.
- **PBT-07/PBT-08**: constrained generators, shrinking and seed `20260715` are
  active.
- **PBT-09**: pytest plus Hypothesis remains the existing supported stack.
- **PBT-10**: generated properties complement 54 real HPR calls, six real
  convergence sentinels, direct process MVR and focused defect regressions.

No blocking Property-Based Testing finding exists. Security and Resiliency
extensions remain disabled and were skipped.

## Limitations

- No global optimum is claimed; only material convergence to the best viable
  objective observed during each bounded sentinel run.
- No universal fluid or process feasibility is claimed; typed failures are
  valid public robustness outcomes.
- Runtime observations are diagnostic, not an SLA or timing assertion.
- Execution is sequential because the observation wrapper is process-global.
- Full repository Build and Test remains the next workflow stage; Code
  Generation verification covers the complete affected surface.
