# HPR and MVR Benchmark E2E Build and Test Summary

## Build Status

- **Build tool**: `uv` 0.11.29 with Hatchling on CPython 3.14.2.
- **Build status**: Success.
- **Documentation**: warning-strict Sphinx build succeeded for 57 sources.
- **Wheel**: `dist/openpinch-0.6.9-py3-none-any.whl`.
- **Source distribution**: `dist/openpinch-0.6.9.tar.gz`.
- **Wheel SHA-256**:
  `13d87ffc073715a1b972aaf5f9faa0dd4e52315fe173f9a41279ac44d9c0d66a`.
- **Source distribution SHA-256**:
  `5ca55a92d394ce4d78224906414d9d2962b4c540bb041178bd5236c3a03d0c89`.
- **Isolated wheel smoke**: passed import, packaged `process_mvr.json`
  construction, and installed CLI help outside the source checkout.

## Test Execution Summary

### Unit and Property Tests

- 11 benchmark-helper and Hypothesis property tests passed.
- Duplicate-temperature alignment, equivalent-row collapse, and
  conflicting-row rejection regressions passed within the full suite.
- Fixed Hypothesis seed: `20260715`.
- Status: Pass.

### End-to-End Tests

- 54 standard HPR benchmark assignments passed their outcome, transaction, and
  search-bound contracts.
- One separate direct process-MVR workflow passed component, thermodynamic,
  detachment, serialization, and downstream-targeting checks.
- Six sentinels demonstrated bounded convergence toward the best viable
  observed objective.
- Focused gate: 66 passed in 47.37 seconds.
- Distinct workflows: 55 total, comprising 28 process-level and 27 site-level
  integrations when the separate process-MVR workflow is included.
- Status: Pass.

### Complete Integration and Regression Suite

- Collected: 3,450.
- Selected: 3,446.
- Passed: 3,443.
- Skipped: three optional notebook execution profiles not enabled for the
  normal gate.
- Deselected: four external-solver tests.
- Failures: zero.
- Duration: 653.22 seconds.
- Branch-aware coverage: 96%, exceeding the 95% requirement.
- Status: Pass.

### Performance and Convergence

- All profile-specific selected-cycle and public-observation caps passed.
- All public HPR calls remained below 50 observed search evaluations.
- All six sentinels showed strict objective improvement and selected the best
  viable observed value within tolerance.
- Wall-clock results are informational; no portable latency SLA was approved.
- Global optimality is not claimed.
- Status: Pass.

### Static, Packaging, and Contract Checks

- Ruff lint: Pass.
- Changed-file format check: Pass, nine files.
- Python compilation: Pass.
- Patch whitespace validation: Pass.
- Warning-strict documentation: Pass.
- Wheel and source distribution: Pass.
- Installed-wheel import/resource/CLI smoke: Pass.
- Public HPR and detached-result contracts: Pass within the complete suite.

## Outcome Classification

The 54-case HPR matrix produced 40 solved outcomes, five legitimate no-ops,
nine bounded typed failures, and no unexpected exceptions. These counts are
specific to the verified environment. The benchmark accepts typed failure as a
robust bounded outcome but requires every designated convergence sentinel to
solve successfully.

## Extension Compliance

- **Property-Based Testing**: Compliant. Seeded generated properties cover
  assignment, classification, state atomicity, diagnostics, observer behavior,
  convergence, repeated points, shrinking, and tolerance boundaries alongside
  real examples.
- **Security Baseline**: N/A because the extension is disabled for this
  workflow. No security claim is made.
- **Resiliency Baseline**: N/A because the extension is disabled for this
  workflow. The approved benchmark robustness contracts nevertheless passed.

## Validation Notes

The final isolated-wheel command passed. Earlier temporary-environment attempts
were harness setup errors: the temporary shell did not initially expose `uv`, a
deliberate `--no-deps` installation omitted pandas, and one post-construction
probe referenced an unsupported convenience attribute. None reached or exposed
a product defect; each was corrected without changing application code.

All generated Markdown uses ordinary headings, lists, tables, and fenced shell
commands. It contains no Mermaid or ASCII diagram requiring a diagram parser.

## Overall Status

- **Build**: Success.
- **Tests**: Pass.
- **Coverage**: Pass.
- **End-to-end robustness**: Pass.
- **Bounded convergence evidence**: Pass.
- **Ready for Operations review**: Yes. The Operations phase is a placeholder,
  and no deployment or publication action is implied.
