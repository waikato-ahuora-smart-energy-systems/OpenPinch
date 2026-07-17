# Serialized HEN Target Input Implementation Summary

## Outcome

`TargetInput` now accepts a serialized heat exchanger network through an
independent `HeatExchangerNetworkSchema` hierarchy. A default runtime
`HeatExchangerNetwork.model_dump(mode="json")` mapping is reproduced exactly by
`TargetInput.model_dump(mode="json")["network"]` and after a complete input JSON
round trip.

## Implemented Changes

- Removed `HeatExchangerStreamRole` and migrated runtime endpoint roles,
  extraction, fake execution, controllability, fixtures, and generated cases to
  `StreamID`.
- Added transport schemas for network, exchanger, period state, and area slice
  records in `OpenPinch.contracts.input`.
- Added optional `TargetInput.network` without domain-object conversion,
  endpoint-to-input-stream cross-validation, or implicit synthesis seeding.
- Mirrored runtime identity, direction, period ordering, numeric, split,
  multiperiod alignment, and area consistency invariants.
- Excluded runtime-private metadata from the transport field set so nested
  unknown-field validation rejects it.
- Added example, mutation, drift, architecture, and seeded property-based tests.
- Documented the canonical JSON mapping bridge and title-case endpoint values.

## Contract Decisions

- Default runtime mappings have exact parity.
- Sparse dumps using `exclude_none`, `exclude_defaults`, or `exclude_unset` are
  accepted; a later canonical dump can restore defaults.
- The encoded string from `model_dump_json()` is not a valid nested network;
  callers must decode it first.
- `network` may be absent, `null`, or an empty network.
- `StreamID.Unassigned`, lowercase legacy endpoint values, and kind/role
  mismatches are rejected.
- `StreamID` is string-backed, so default Python-mode runtime and transport
  dumps remain directly JSON-serializable while retaining enum typing.

## Extension Compliance

- Security: N/A, disabled in extension configuration.
- Resiliency: N/A, disabled in extension configuration.
- Partial property-based testing: compliant. The generated aligned-network
  property uses seed `20260717`, 30 bounded examples, normal Hypothesis
  shrinking, CI-discoverable test placement, and invariant-based round-trip and
  ordering assertions.
