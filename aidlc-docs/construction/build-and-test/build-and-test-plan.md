# RTD and Comprehensive HPR Notebook Build and Test Plan

## Applicability Assessment

- Build verification is applicable because the generated notebook and RTD data
  ship in the source and wheel distributions.
- Unit and property tests are applicable to generator reproducibility, notebook
  source policy, map serialization, target records, fluids, and physical map
  invariants.
- Integration and end-to-end tests are applicable to the public HPR
  target-record-map-export boundary, notebook execution, RTD, resources, and
  packaging.
- Performance checks are applicable to notebook runtime and the existing
  300-second real TESPy public target-to-map smoke.
- Network/concurrent-user load testing is N/A because OpenPinch is a local
  Python library.
- Security tests: N/A because the Security extension is disabled and the change
  introduces no authentication, network service, or secret boundary.

## Execution Checklist

- [x] Analyze build, unit, property, integration, end-to-end, performance,
  documentation, packaging, and extension requirements for the focused
  follow-up.
- [x] Refresh the canonical build, unit, integration, and performance
  instructions for notebook 09 and RTD verification.
- [x] Build fresh source and wheel artifacts and verify that notebook 09 is
  packaged; verify repository-owned RTD/tutorial inventory through its own
  Sphinx and test gates.
- [x] Run the complete repository regression with Hypothesis seed `20260715`,
  plus warning-strict Sphinx, Ruff, formatting, compilation, and patch hygiene.
- [x] Execute notebook 09 in a clean directory and retain the successful bounded
  real TESPy public target-to-map test as the engine oracle.
- [x] Generate the final build-and-test summary, update state/audit, and present
  the standardized review prompt.

## Extension Compliance

- Property-Based Testing: enabled; generated, invariant, round-trip, ordering,
  lifecycle, cache, and oracle tests remain blocking.
- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.

This plan contains no Mermaid or ASCII diagram. Its Markdown headings,
checkboxes, paths, identifiers, and special characters were validated before
update.
