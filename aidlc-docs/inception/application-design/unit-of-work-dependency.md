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
