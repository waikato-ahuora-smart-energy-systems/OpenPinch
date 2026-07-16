# Private Helper Reorganization Requirements

## Status

Approved for implementation by the user's explicit implementation request.

## Intent

Reorganize `OpenPinch.classes` private helpers into owner-oriented packages,
extract cohesive non-core responsibilities from the largest public class
modules, and make the stream and heat-exchanger runtime records parent-owned
implementation details.

## Requirements

- `StreamSegment`, `HeatExchangerPeriodState`, and
  `HeatExchangerAreaSlice` retain their internal class names but have no public
  aliases, barrel exports, documentation promises, or compatibility shims.
- `StreamSegmentSchema` remains a public input schema.
- `Stream` accepts mappings, schema-like objects exposing `model_dump()`, and
  internal segment records, and returns immutable segment tuples.
- `HeatExchanger` accepts mappings for period state and area contribution
  records and returns typed internal records for read-only inspection.
- Public parent APIs, serialized structures, exceptions, ordering, and
  numerical behavior otherwise remain unchanged.
- Private helpers are grouped by their owning public class and responsibility.
- Existing internal services, tests, notebooks, and documentation use private
  record imports only where construction or runtime type checks are required.
- Old public imports and Python pickle paths are intentionally unsupported.

## Scope Decisions

- This is an internal refactor with an intentional public API cleanup; user
  stories, infrastructure design, NFR design, and deployment work add no value
  and are skipped.
- Security and Resiliency extensions remain disabled.
- Partial Property-Based Testing enforcement remains blocking for applicable
  round-trip, normalization, ownership, ordering, and transaction invariants.

## Acceptance Criteria

- The approved owner-package hierarchy exists and retired loose helpers and
  `_problem`/`_workspace` packages do not.
- Public barrels and modules do not expose the three private runtime records.
- Construction, serialization, copy/pickle, ownership, rollback, Pydantic
  schema, and interval invariants are covered by deterministic and generated
  tests using seed `20260715` where applicable.
- Focused tests, complete non-solver and available solver tests, Ruff,
  warning-free Sphinx, notebook parsing, package builds, stale-path searches,
  API checks, and `git diff --check` pass.

