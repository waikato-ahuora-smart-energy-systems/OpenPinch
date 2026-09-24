# Unit 2 Domain Entities

## Prepared CoolProp stage capability

A frozen private value containing topology, stage ordinal, role, source fluid,
evaporating/suction temperature, and condensing/discharge temperature. It owns
no engine object.

## Prepared CoolProp targeting

An ordered frozen tuple of prepared stage capabilities for one topology. Its
stage ordinals are contiguous and its count matches normalized topology inputs.

## Candidate cache

A call-local mapping from an exact finite coordinate tuple to one search-mode
`HPRBackendResult` and scalar objective. It is discarded after the request.

## Failure accumulator

A call-local count by `HPRFailureCategory` plus no more than 16
`HPRFailureDiagnostic` representatives. It finalizes to `HPRFailureSummary`.

## Resolved search request

The existing HPR input plus one `HPRSearchBudget`; no persisted state or database
entity is introduced.
