# RTD and Comprehensive HPR Notebook Code Summary

## Outcome

The existing notebook 09 is now the canonical comprehensive HPR
target-to-performance-map tutorial. Its generator, packaged notebook, RTD
guidance, tutorial manifest, and regression contracts are synchronized. No
production API, schema, solver dependency, root export, or thermodynamic
calculation changed.

## Modified Application and Documentation Files

- `scripts/generate_tutorial_notebooks.py`: expands notebook 09 into a coherent
  default-CoolProp target, winning-record, map-request, three-point part-load
  map, plain-JSON export, explicit-TESPy mixture, and comparison study.
- `OpenPinch/data/notebooks/09_vapour_compression_and_brayton.ipynb`:
  regenerated canonical packaged notebook with the `slow-hpr` profile.
- `docs/guides/heat-pump-workflows.rst`: identifies notebook 09 as the
  comprehensive executable example and explains the exact target-to-map and
  no-fallback boundary.
- `docs/examples/notebook-series.rst`: publishes the expanded notebook scope.
- `docs/examples/tutorial-coverage-map.rst`: restores 197/197 executable
  code-cell mapping coverage and its counting rule.
- `docs/_data/tutorial-coverage.csv`: restores
  `problem.target.hpr_performance_map` to `mapped and executable`.
- `tests/packaging/test_notebooks.py`: requires the complete notebook 09 study
  and narrowly permits the documented specialist request-contract import.
- `tests/packaging/test_tutorial_coverage.py`: requires every manifest operation
  to use an accepted notebook execution status and checks exact 197/197 RTD
  alignment.
- `tests/packaging/test_docs_consistency.py`: requires RTD and notebook-series
  ownership statements for the comprehensive example.

## Tutorial Narrative and Boundary

The notebook uses `PinchProblem` as the workflow root and imports only the
published specialist `HprPerformanceMapRequest` contract. Omitting
`simulation_backend` demonstrates the CoolProp default. A successful target's
detached `target_simulation_record` supplies the source and sink temperatures
for a map request at load fractions 0.50, 0.75, and 1.00. The returned schema is
serialized with `model_dump(mode="json")` for a downstream package such as
OpenUtility, which does not need to import OpenPinch.

TESPy is shown as a separate explicit selection using
`HEOS::R32[0.5]&R125[0.5]`, one evaporator, one condenser, and the supported
single-stage boundary. Pure-fluid and provider-registered blend forms are also
shown. Installed property support and the requested operating state determine
mixture feasibility. A guarded failure remains attributed to the selected
backend; it neither falls back to CoolProp nor produces a partial map.

## Verification Evidence

- RED phase: four intended missing notebook/RTD failures and one passing live
  public-inventory contract.
- Tutorial, RTD, and resource gate: 56 passed and 3 optional-profile skips.
- Focused HPR contract, target, record, basis, accessor, map, mixture,
  serialization, PBT, and real-TESPy gate: 302 passed with Hypothesis seed
  `20260715`.
- Notebook 09: all five code cells compiled and executed from a clean temporary
  directory in 27.716 seconds.
- Successful real-engine oracle: the focused suite's bounded public TESPy
  target-to-map smoke generated two ordered map points within the 300-second
  budget.
- Generator: repeated final passes preserved SHA-256
  `8c9462c12d12678514900a78beabdd31db73adb266b5d6d4fcbacabe1518c1ba`,
  proving byte idempotence.
- Documentation: Sphinx HTML build passed with warnings treated as errors.
- Static quality: repository Ruff lint, changed-surface Ruff formatting, Python
  compilation, and `git diff --check` passed.

## Requirement Traceability

- FR-NB-01: existing notebook 09 and its generator were modified in place.
- FR-NB-02: omitted/default CoolProp and explicit TESPy selectors are present
  with bounded one-by-one topology settings.
- FR-NB-03: pure, registered-blend, and explicit molar-mixture forms are shown
  with feasibility guidance.
- FR-NB-04: winning record, derived coordinates, request, map call, and multiple
  part-load points form one conditional executable sequence.
- FR-NB-05: the versioned map is exported as plain JSON and the OpenUtility
  consumer boundary is explicit.
- FR-NB-06: guarded numerical and optional-engine failures have no fallback and
  no partial-map behavior.
- FR-NB-07: the HPR guide, notebook series, and coverage page are synchronized.
- FR-NB-08: the manifest again records notebook-executable map coverage.
- NFR-NB-01 through NFR-NB-03: idempotence, compilation, clean execution, small
  grids, bounded searches, and the real-engine oracle passed.
- NFR-NB-04 through NFR-NB-06: warning-strict RTD, package separation, no public
  compatibility change, seeded PBT, lint, formatting, compilation, and patch
  hygiene passed.

## Known Runtime Behavior

The packaged sample's real numerical screening can return no feasible CoolProp
or TESPy placement on a particular run. Notebook 09 handles that as an
engineering outcome, completes all cells, exports no partial map, and labels the
backend result accurately. The deterministic contract tests and bounded real
TESPy public smoke separately prove the successful target-to-map branch.

## Extension Compliance

- Property-Based Testing: compliant. Existing map physical properties,
  serialization round trips, target-record invariants, lifecycle behavior,
  generator idempotence, and the real-engine oracle passed with the required
  seed.
- Security Baseline: N/A because the extension is disabled and this follow-up
  adds no security surface.
- Resiliency Baseline: N/A because the extension is disabled and this follow-up
  adds no runtime service or infrastructure.

This summary contains no Mermaid, ASCII diagram, or embedded JSON/YAML block.
Markdown structure, paths, identifiers, code literals, and special characters
were validated before creation.
