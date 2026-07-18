# Post-Implementation Import and Type Fixes Code Generation Plan

## Approved Scope

Fix the five findings from the package-wide unresolved-import and type audit
without changing `OpenPinch.main.pinch_analysis_service` or unrelated dynamic
solver/model behavior.

## Execution Checklist

- [x] Step 1: Reproduce and classify the two unresolved type imports, Zone
  self-import, total-site keyword mismatch, and crossflow row failure.
- [x] Step 2: Record the user's explicit fix approval and this focused plan.
- [x] Step 3: Add regressions for resolvable internal type imports, no
  TYPE_CHECKING self-import, canonical total-site period forwarding, and invalid
  crossflow row validation.
- [x] Step 4: Correct the type-only import owners and remove the Zone self-import.
- [x] Step 5: Correct total-site period forwarding and validate crossflow row
  counts explicitly.
- [x] Step 6: Run focused architecture, total-site, heat-transfer, import-sweep,
  Ruff, and patch-hygiene gates.
- [x] Step 7: Run the complete non-solver suite and finalize AI-DLC evidence.

## Extension Compliance

- Security: disabled, therefore N/A.
- Resiliency: disabled, therefore N/A.
- Partial Property-Based Testing: N/A because these fixes correct import
  ownership, argument naming, and input validation without changing numerical
  algorithms.
