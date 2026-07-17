# Serialized HEN Target Input Code Generation Plan

This checklist is approved by the user's explicit implementation request and
is the single source of truth for the implementation.

- [x] Step 1: Re-read the current package, confirm a clean baseline, and capture requirements and design artifacts.
- [x] Step 2: Replace HeatExchangerStreamRole with StreamID in the domain model and all production consumers.
- [x] Step 3: Add the four independent transport schemas and TargetInput.network.
- [x] Step 4: Migrate test fixtures and add example-based validation and exact dump-parity tests.
- [x] Step 5: Add seeded property-based round-trip coverage and drift guards.
- [x] Step 6: Update API/HEN documentation and implementation evidence.
- [x] Step 7: Run focused HEN and contract verification.
- [x] Step 8: Run the complete non-solver, Ruff, Sphinx, and patch-hygiene gates.
- [x] Step 9: Finalize Build and Test evidence and review handoff.
