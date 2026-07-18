# Serialized HEN Target Input Requirements

## Intent Analysis

- **Request type**: Public input-contract enhancement and intentional enum migration.
- **Scope**: HEN domain records, input contracts, HEN consumers, tests, and documentation.
- **Complexity**: Moderate because the JSON contract and runtime enum must change together.
- **Compatibility policy**: Clean break for the unsupported `HeatExchangerStreamRole` enum.

## Functional Requirements

- `TargetInput` shall accept an optional `network` mapping produced by
  `HeatExchangerNetwork.model_dump(mode="json")`.
- The mapping shall validate into independent input-contract schemas rather
  than a runtime `HeatExchangerNetwork` instance.
- A default JSON-mode dump of the validated `TargetInput` shall reproduce the
  supplied canonical network mapping exactly.
- HEN endpoint classifications shall use `StreamID.Process` and
  `StreamID.Utility`; `StreamID.Unassigned` and former lowercase role values
  shall be rejected.
- `HeatExchangerStreamRole` shall be removed with no compatibility alias.
- Private solver and source metadata excluded by runtime serialization shall
  also be absent from and rejected by the transport contract.
- Existing inputs without `network`, and inputs with `network: null`, shall
  retain their current behavior.

## Validation Requirements

- Transport schemas shall mirror runtime identity, direction, period, finite
  number, split-fraction, and area-consistency invariants.
- Network stream names shall not be cross-validated against input streams or
  utilities in this change.
- The network shall be retained by canonical input preparation but shall not
  automatically seed or execute HEN synthesis.

## Quality Requirements

- Example-based tests shall cover all exchanger kinds, invalid StreamID
  combinations, metadata exclusion, and exact dump parity.
- A seeded Hypothesis test shall verify runtime-dump to TargetInput JSON
  round-trips using domain-specific generated exchangers.
- Focused HEN and contract tests, the complete non-solver suite, Ruff, and
  warning-free Sphinx shall pass.

## Approval

The user's explicit `PLEASE IMPLEMENT THIS PLAN` request approves these
requirements and the dependency-ordered implementation sequence.
