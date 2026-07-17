# User Stories

## US-HEN-INPUT-1: Supply a Serialized Network

As a programmatic HEN user, I want to pass the mapping returned by
`HeatExchangerNetwork.model_dump(mode="json")` as `TargetInput.network`, so
that I can persist and reload a network through the standard input contract.

### Acceptance Criteria

- The mapping validates into `HeatExchangerNetworkSchema`.
- The canonical nested mapping is reproduced exactly by TargetInput JSON-mode
  serialization and by a complete JSON round-trip.
- Endpoint classifications serialize as `Process` and `Utility` using
  `StreamID`.
- Unassigned, legacy lowercase, and invalid endpoint combinations fail
  validation.
- Private metadata remains absent and is rejected when hand-supplied.
- Omitting the network remains backward compatible.

## US-HEN-INPUT-2: Detect Contract Drift

As an OpenPinch contributor, I want tests that compare actual runtime JSON
dumps with the transport schema, so that new JSON-visible HEN fields cannot be
silently discarded.

### Acceptance Criteria

- Runtime and transport dump keys are compared at network and exchanger levels.
- Excluded metadata fields are explicitly accounted for.
- Generated valid networks satisfy the round-trip property under Hypothesis.
