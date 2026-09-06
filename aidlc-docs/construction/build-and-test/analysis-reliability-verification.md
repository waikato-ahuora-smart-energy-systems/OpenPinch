# Analysis reliability and extensibility verification

Verified 2026-09-05T22:47:11Z on macOS arm64, Python 3.14.2.

## Implemented scope

All four approved units are complete: execution snapshots and invalidation;
targeting corrections and period chaining; method/provenance/result adapters;
aggregation policies and documentation. New scientific methods were excluded.
Brayton is unavailable and fails before preparation. No legacy aliases were added.

## Final checks

| Check | Result |
| --- | --- |
| Configured full suite, Hypothesis seed 20260715, excluding solver marker | 3,086 passed; 3 expected skips; 4 solver deselections |
| Coverage with branch tracking, required combined threshold 95% | 95.73% combined; passed |
| Statement coverage | 97.11% |
| Branch coverage | 91.25% |
| Real-solver marker gate | 3 passed; existing nine-stream benchmark skipped |
| TESPy marker gate | 58 passed |
| Guarded public TESPy target-to-map smoke | Passed |
| Ruff and patch whitespace checks | Passed |
| Sphinx with fail-on-warning and keep-going | Passed |
| Tutorial inventory and generated notebook consistency | Passed |
| Base-profile notebook execution | Passed within the full suite |
| Affected process-MVR/cascade notebook 11 | Executed separately; passed |
| Wheel and source distribution | Built successfully |
| Final wheel installed in isolated core and TESPy environments | Both smoke tests passed |

The three normal-suite skips are the explicitly opt-in slow-HPR, solver, and
interactive notebook profiles. All changed base-profile notebooks ran, and the
changed slow-HPR notebook 11 ran separately. The existing nine-stream solver
benchmark remains explicitly disabled by its repository test marker. No new
skips or coverage exclusions were introduced for this implementation.

## Acceptance evidence

- `test_analysis_reliability.py` covers process-MVR recovery/work agreement at
  different worker counts, canonical period rows, ambient-temperature overrides,
  local resolution of foreign zones, scratch child prerequisites, retained-period
  exergy without thermal recomputation, transactional failures, and invalidation.
- `test_analysis_lifecycle.py` uses a shrinking Hypothesis state machine for
  component lifecycle, configuration rebuilding, snapshots, and serialization.
  Generated inputs exercise worker equivalence, batch rollback, and provenance.
  Snapshot checks verify component ownership and shared stream identity.
- `test_analysis_contracts.py` checks immutable provenance, foreign/stale base
  references, public/workspace/catalog/documentation parity, specialist adapters,
  temporary-configuration fingerprint stability, and named multiperiod HEN identity.
- `test_analysis_policies.py` rejects undeclared and unimplemented aggregation
  policies, tests every declared numerical policy, preserves period identity,
  and excludes per-period simulation records from weighted rows.
- Existing service, hierarchy, design, solver, reporting, packaging, and
  architecture tests passed with expectations updated for detached observations.

Regressions were demonstrated before corrections: 11 execution failures, four
initial method/provenance failures, two aggregation failures, plus the final
fingerprint and multiperiod-design identity regressions. Later full-suite failures
were resolved without disabling property testing or reducing test coverage gates.

## Intentional contract changes

Observation properties return detached snapshots or read-only mappings. Component
activation/deactivation remain explicit supported mutations. Input rebuilding
continues to discard memory-only components; recreate them on rebuilt inputs.
Returned analyses are detached, and base-target references must match current
owner, scope, period, and input provenance. Selected-period reports omit stale
rows; scalar changes supersede a published all-period report. Weighted rows no
longer claim the first period's HPR simulation record as aggregate evidence.

Contributor and migration guides, the capability catalog/table, tutorial inventory,
affected notebooks, and release notes were updated together. At verification
completion, source changes were uncommitted. Subsequent commit authorization is
recorded in the audit log. Nothing was published or deployed.

## Extension compliance

- Property-Based Testing: compliant; enabled, using Hypothesis with shrinking and
  repository seed 20260715. Existing test-specific seeds were retained.
- Security: disabled configuration unchanged; not enforced.
- Resiliency: disabled configuration unchanged; not enforced.
- Operations/deployment: not applicable to the approved library implementation.

## PR 96 review corrections

The user authorized fixing all three confirmed Codex findings. The added
regression baseline produced six failures and two passes before correction.

- Published rows and graphs now match the selected zone and traversal. The
  connected prepared tree is retained so compatible enrichment can reuse it.
- HPR families include their effective simulation backend in provenance settings.
  Generated heat-pump and refrigeration backend transitions reject superseded
  references, normalize backend spelling, and preserve failed-call atomicity.
- Performance-map provenance observations are detached recursively. Mutating
  nested dictionaries/lists cannot change validated state or exported JSON;
  the schema 1.0 resource remains byte-for-byte consistent with its generator.

Hypothesis tests generate scope-switch and backend-transition sequences, plus
structured JSON payloads. They retain shrinking and seed 20260715. The backend
transition tests replace only the scientific service with a deterministic result;
the public executor, provenance, base-reference validation, and exergy paths run.
The real TESPy public target/map smoke separately validates solver integration.

| Correction check | Result |
| --- | --- |
| Application, contracts, service orchestration, energy transfer | 634 passed |
| New regressions and generated properties | Passed |
| Real public TESPy target/map smoke | 1 passed |
| Real-solver gate | 3 passed, 1 existing benchmark skip |
| Ruff, changed-file format checks, patch whitespace | Passed |
| Warning-strict Sphinx | Passed |
| Tutorial/docs inventory and committed map schema consistency | 25 passed |
| Wheel and source distribution | Built successfully |
| Installed core and TESPy wheel smoke tests | Both passed |
| Installed core nested provenance isolation and JSON round trip | Passed |
| Full configured suite, Hypothesis seed 20260715, excluding solver marker | 3,094 passed, 3 existing skips, 4 solver deselections |
| Coverage with branch tracking, combined threshold 95% | 95.73%; passed (97.11% statement, 91.25% branch) |

The full run completed in 482.89 seconds, including base notebook execution and
the TESPy-marked tests. No new skips, exclusions, or suppressed property checks
were introduced. Local logs are `/tmp/openpinch-review-full.log`,
`/tmp/openpinch-review-coverage.json`, `/tmp/openpinch-review-solvers.log`,
`/tmp/openpinch-review-tespy-smoke.log`, `/tmp/openpinch-review-docs.log`,
and `/tmp/openpinch-review-artifact-{core,tespy}.log`.

No scientific algorithm, capability, or tutorial call signature changed. Existing
notebooks and inventory remain applicable; their consistency checks passed.
Migration, contributor, and release notes describe the corrected behavior.
Security and Resiliency remain disabled; PBT remains enabled. No deployment or
GitHub review reply was performed. Corrections are local and uncommitted.

## PR 96 CI fixture version correction

The new Hypothesis property first failed for explicit release 0.6.5 because both
golden JSON examples embedded the installed release. Their historical generating
version is now fixed at 0.6.4; neither committed JSON nor real-map runtime
provenance changes. Generated version tuples retain shrinking and seed 20260715.

- Contract suite plus resource and documentation consistency checks: **207 passed**
  in 3.12 seconds (`/tmp/openpinch-ci-fixture-tests.log`).
- Original failing CI test also passes with distribution metadata patched to 0.6.5.
- Authoritative generator `--check`, Ruff lint/format and patch whitespace pass.
- Warning-strict Sphinx passes (`/tmp/openpinch-ci-fixture-docs.log`).

This narrow generator correction does not rerun the earlier full numerical,
notebook, solver, coverage or artifact gates; their previous evidence remains
above. Security and Resiliency remain disabled; PBT remains enabled.
