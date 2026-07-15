# Segmented Variable-CP Stream Requirements

## Intent Analysis

- **Request type**: Cross-cutting domain-model enhancement and solver refactor.
- **Scope**: System-wide across schemas, streams, targeting, thermal components, HEN synthesis, reporting, and persistence.
- **Complexity**: High because existing numerical services assume one constant heat-capacity flowrate per physical stream.
- **Goal**: Model one physical variable-CP stream as one parent `Stream` with ordered `StreamSegment` children.

## Functional Requirements

- FR-01: Add a public `StreamSegment(Stream)` type while retaining the graph schema named `Segment`.
- FR-02: A segmented parent exposes immutable ordered segments and parent-level identity.
- FR-03: Parent supply, target, duty, effective CP, and effective HTC are derived from segments.
- FR-04: Segment order is preserved; no service may auto-sort it.
- FR-05: For every period, `segment[i].t_target` must equal `segment[i+1].t_supply` within the configured thermal tolerance.
- FR-06: Discontinuous, overlapping, reversed, or direction-inconsistent profiles are rejected atomically.
- FR-07: Nested explicit segments and temperature-cumulative-heat profiles are supported by structured inputs.
- FR-08: Profile values are authoritative; duplicated parent aggregates are validation assertions.
- FR-09: Multiperiod calculated profiles use a stable union breakpoint grid.
- FR-10: Existing flat input rows remain independent streams.
- FR-11: Targeting kernels expand segments thermodynamically but count parents topologically.
- FR-12: HPR and MVR services emit one parent per physical thermal duty or compression stage.
- FR-13: HEN axes, matches, positions, and unit counts remain parent based.
- FR-14: HEN heat balances use parent cumulative heat coordinates with piecewise temperature mappings.
- FR-15: Reported and verified HEN area is the sum of ordered duty-aligned
  segment-pair slices. Topology optimization retains the smooth Chen LMTD
  surrogate, followed by exact segment-resolved post-processing.
- FR-16: Parent-level reporting remains the default; expanded reporting is opt-in.
- FR-17: Copies, pickles, workspace bundles, and caches preserve segment order and period context.

## Non-Functional Requirements

- NFR-01: Existing constant-CP inputs and flat stream behavior remain backward compatible.
- NFR-02: Equivalent flat and segmented thermal profiles produce matching targets within numerical tolerance.
- NFR-03: Invalid mutations roll back without exposing partially updated parent state.
- NFR-04: Segment revisions invalidate cached numeric views.
- NFR-05: Solver paths must not silently substitute aggregate CP for a segmented profile.
- NFR-06: Existing stream counts and HEN topology size depend on parents, not segment count.
- NFR-07: Property-based tests use Hypothesis with domain generators, shrinking, and reproducible seeds.

## Success Criteria

- One variable-CP physical stream occupies one collection entry and has multiple ordered segments.
- Adjacent boundary continuity is enforced in every period.
- Direct and indirect targets match equivalent legacy flat profiles.
- HEN results retain one exchanger per parent match and expose segment area contributions summing to the exchanger area.
- The complete supported test suite passes without aggregate-CP use in segmented thermal equations.
