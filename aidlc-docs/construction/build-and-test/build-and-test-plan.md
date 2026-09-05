# HPR Performance Map Build and Test Plan

## Applicability Assessment

- Unit tests: applicable to contracts, fluid resolution, simulator adapters,
  target integration, records, basis conversion, public accessors, and period
  propagation.
- Property tests: applicable to schema round trips, physical closure, mixture
  fidelity, selector determinism, cache identity, lifecycle, ordering, and
  failure isolation.
- Integration and contract tests: applicable across Units 1 through 3, current
  HPR targeting, reporting, resources, documentation, and package artifacts.
- End-to-end tests: applicable to public CoolProp and explicit TESPy target to
  map workflows from both the checkout and installed wheel.
- Performance tests: applicable to large fake maps, exact-cache memory and call
  bounds, repeated cleanup, and the guarded real TESPy workflow.
- Security tests: N/A because the Security extension is disabled and the change
  introduces no authentication, network service, or secret boundary.
- External solver tests: applicable as the repository-wide regression gate;
  HPR map generation itself introduces no solver dependency.

## Execution Checklist

- [x] Analyze unit, property, integration, contract, end-to-end, performance,
  documentation, packaging, and external-solver requirements.
- [x] Refresh the canonical build, unit, integration, and performance
  instructions for the HPR delivery.
- [x] Run Ruff, changed-surface formatting, compilation, patch hygiene, and the
  warning-strict documentation build.
- [x] Run the complete repository suite with fixed Hypothesis seed `20260715`
  and configured external solvers.
- [x] Run the dedicated real-TESPy and 300-second public target-to-map profile.
- [x] Verify fresh source/wheel archives and isolated core/TESPy wheel smokes.
- [x] Generate the final build-and-test summary and update workflow state/audit.
- [x] Present the standardized Build and Test review prompt.

## Extension Compliance

- Property-Based Testing: enabled and applicable throughout the unit and
  integration gates.
- Security: disabled; N/A.
- Resiliency: disabled; N/A.

This plan contains no Mermaid or ASCII diagram. Its Markdown headings,
checkboxes, inline paths, and code identifiers were validated before update.
