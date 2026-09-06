# Analysis reliability and extensibility implementation

The user approved implementation of the complete four-unit plan on 2026-09-06.
New scientific methods are excluded. Existing method-specific return types remain;
targeted observation/provenance API changes are authorized.

## Execution checklist

- [x] 1. Inspect execution, component ownership, reporting, and workflow rules.
- [x] 2. Add and run failing execution/state regressions (11 expected failures).
- [x] 3. Implement shared snapshots, atomic execution, and invalidation.
- [x] 4. Correct period provenance, configuration, zone ownership, traversal,
  and retained-period chaining; verify the execution regressions.
- [x] 5. Add failing method/provenance/adapter contract tests (4 expected failures).
- [x] 6. Implement immutable capability metadata, provenance, family adapters,
  and early unsupported-method checks; verify contracts.
- [x] 7. Add failing aggregation-policy and lifecycle properties (2 expected failures).
- [x] 8. Implement explicit metric policies and contributor/migration docs;
  synchronize public inventory and affected tutorial sources.
- [x] 9. Run full tests, coverage, real solver/TESPy checks, lint, documentation,
  notebooks, distribution builds, and installed-artifact smoke tests.
- [x] 10. Record results, limitations, and completed checkboxes in AI-DLC state.

## Design and acceptance

Execution copies the root zone and components in one deepcopy memo, retaining
internal stream identity and rebinding component owners. All-period execution
uses the same worker function at every worker count and commits complete batches.
One prepared snapshot per period supports enrichment without repeated solvers.
Failed work must leave the prior successful state unchanged.

Regressions cover MVR serial/parallel equality, mixed-period reporting, exergy
ambient overrides, local resolution of foreign zones, selected-zone traversal,
dependent period chains, atomic failures, and component cache invalidation.
Contract checks cover provenance, nonthermal adapters, complete metric policies,
catalog/accessor parity, and early Brayton rejection.

Property testing covers round trips, isolation, lifecycle sequences, period
identity, and aggregation invariants, using Hypothesis seed 20260715 with
shrinking. PBT is enabled; Security and Resiliency remain disabled. Deployment
and additional scientific algorithms are not part of this implementation.

## PR 96 review corrections

The user authorized these three investigated corrections with "fix them".
They preserve the approved execution, provenance, and JSON contracts.

- [x] 11. Add failing regressions for sibling-scope period batches, backend
  replacement identity, and nested performance-map provenance mutation.
- [x] 12. Scope published rows/graphs to the selected zone and requested
  descendants while retaining compatible prepared prerequisites.
- [x] 13. Include the effective HPR simulation backend in family-owned
  provenance settings; reject references superseded by a backend change.
- [x] 14. Detach map provenance observations, preserving schema 1.0 and
  JSON serialization, copying, and equality behavior.
- [x] 15. Run regression/property, broader application/contract, TESPy, lint,
  documentation, and artifact checks; record results and migration notes.

Properties: generated scope-switch sequences preserve selected-scope output
and same-scope enrichment without thermal recalculation; backend transitions
invalidate superseded references; nested JSON observation mutations leave the
validated map unchanged and serialization round trips preserve equality.
Use Hypothesis shrinking and seed 20260715. Security and Resiliency remain
disabled. Deployment and GitHub review replies are outside this correction.

## PR 96 CI fixture correction

- [x] 16. Reproduce golden-resource drift with generated installed versions.
- [x] 17. Make fixture provenance deterministic while preserving real-map runtime provenance.
- [x] 18. Verify contract/property, generator, lint, documentation checks and record evidence.
