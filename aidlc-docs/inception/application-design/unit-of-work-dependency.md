# Unit Dependencies

| Unit | Depends on | Delivery order |
|---|---|---:|
| Domain and Input | Existing value, stream, schema, and preparation layers | 1 |
| Targeting and Integration | Domain and Input | 2 |
| Heat Exchanger Network | Domain and Input; Targeting and Integration | 3 |

## Package Usability Refactor Dependencies

| Unit | Direct dependencies | Why |
|---|---|---|
| 1. Contract and Correctness Foundation | approved requirements and design | Defines the contract and regression evidence consumed by every later unit. |
| 2. PinchProblem Interaction, Targeting, and Configuration | Unit 1 | Implements the frozen problem-level contract and state primitives. |
| 3. Components, Design, Workspace, and Presentation | Units 1 and 2 | Mirrors the problem vocabulary and consumes its resolver/state model. |
| 4. Capability-Complete Tutorial Suite | Units 1, 2, and 3 | Teaches only completed canonical workflows and return views. |
| 5. Documentation and Executable Quality Gates | Units 1, 2, 3, and 4 | Publishes and verifies the final live API and executable tutorial corpus. |

The graph is acyclic and has one implementation order: 1, 2, 3, 4, 5. Units
are logical construction boundaries inside one distributable package, not
independently deployable services. A later unit may reveal a defect in an
earlier contract, but the correction is applied to the owning earlier unit and
its regression tests before downstream work continues.

## Repository Issue Remediation Dependencies

| Unit | Direct dependencies | Downstream consumers | Delivery order |
|---|---|---|---:|
| 1. Application State and Filesystem Contracts | approved requirements and Application Design | Unit 3 documentation and repository gates | 1 |
| 2. Exact OpenHENS Checkout Loading | approved requirements and Application Design | Unit 3 documentation and repository gates | 2 |
| 3. Current Documentation and Drift Guards | final Unit 1 and Unit 2 contracts | final build/test evidence | 3 |

### Update Strategy

- **Approach**: logically parallel Unit 1 and Unit 2 ownership with sequential
  implementation and review for diagnostic clarity.
- **Critical path**: Unit 1, Unit 2, then Unit 3 and repository-wide gates.
- **Coordination points**: architecture/stale-symbol tests and final package
  verification consume both runtime units.
- **Rollback**: each unit is independently revertible; Unit 3 documentation is
  reverted with whichever runtime unit contract is reverted.
- **Deployment**: one wheel/source distribution after all units; no independent
  deployment or version skew is supported.

### Testing Checkpoints

1. Unit 1 focused application, contract, property, and reporting tests.
2. Unit 2 isolated import/cache and comparison prerequisite tests.
3. Unit 3 stale-symbol, architecture, documentation, packaging, and wheel tests.
4. Complete fixed-seed non-solver and repository quality gates after all units.

The graph is acyclic: Units 1 and 2 depend only on approved inception artifacts;
Unit 3 depends on both runtime units; no edge returns to an earlier unit.
