# Remaining data resource relocation

User approved the proposed owner-based resource layout. Continue the existing
package-layout workflow as one minimal refactor; no new business logic, user
stories, infrastructure or performance design is needed. Existing tutorial
relocation edits are preserved. No commit or push is included.

- [x] 1. Inspect consumers and confirm the four JSON resources and destinations.
- [x] 2. Preserve JSON bytes, move contract resources to
  OpenPinch/contracts/resources/hpr_performance_map and the compressor
  characteristic to OpenPinch/analysis/heat_pumps/performance_maps/characteristics.
  Update loaders, generator paths, existing tests and public documentation;
  remove the obsolete OpenPinch/data package and its initializer/cache files.
- [x] 3. Verify contract generation, affected contract/TESPy/resource and packaging
  tests, strict docs, both installed distribution formats, preserved bytes and
  lint. Record results and complete state tracking.

Use the existing Hypothesis contract tests and canonical resource identity checks.
PBT-01 through PBT-06: no new algorithm, transformation or state; unchanged
round-trip/invariant behavior remains covered by existing tests and byte checks.
PBT-07 through PBT-09: use existing domain strategies, normal shrinking and
Hypothesis seed 20260715. Security and Resiliency are disabled and skipped.
