Adding Analysis Methods
=======================

Keep the public API descriptive: add a named method to ``problem.target`` or
``problem.design``. The internal catalog describes capabilities; it is not a
runtime plugin registry or a public string dispatcher.

Contribution checklist
----------------------

1. Declare scopes, prerequisites, period behavior, effective configuration,
   availability, optional dependencies, and result adapter in
   ``application/_problem/targeting/catalog.py``. Fail unsupported methods
   before preparing input state. Add the supported workspace and period mirrors.
2. Keep shared metadata in ``domain.analysis.AnalysisProvenance``. Put the
   method-specific payload in its owning contract or domain family. A
   nonthermal result does not need utility metrics or ``UtilitySummaryTarget``.
   Target families extend ``provenance_settings`` for calculation-affecting
   runtime choices outside configuration. HPR includes its effective simulation
   backend this way, so switching engines invalidates superseded references.
3. Thermal and enrichment accessors use the application execution transaction.
   It owns the local zone address, canonical period, settings, prepared zone
   tree, components, and prerequisites. Snapshot the zone and components
   together. Do not reconstruct runtime snapshots from problem JSON: process
   MVR replacement streams and compressor records are runtime state.
4. Declare target dependencies on the family through ``prerequisite_types``.
   Reuse a prerequisite only when its period and numerical settings match.
   Enrichment should report an actionable missing-prerequisite error; methods
   that compute their prerequisites must do so explicitly. Respect traversal:
   child calculations used only as prerequisites stay in scratch state.
5. Add a family-owned presentation adapter and compose it in the immutable
   adapter mapping. Keep reports, plots, and exports observational. Inverse
   targeting and performance maps keep their specialized non-mutating paths;
   utility placement and HEN retain their shared/specialized solve semantics.
6. Declare every report field with ``report_field`` and a unit/representation
   and aggregation policy: weighted mean, maximum, consensus, derived, or
   exclusion. Implement and test each derived calculation. An undeclared
   field or unimplemented derivation must fail validation.
7. Add example-based regressions before corrections, plus Hypothesis lifecycle,
   snapshot isolation, worker equivalence, provenance, and JSON round trips.
   Preserve shrinking and repository seed ``20260715``. Exercise failures after
   a prior success and verify that the complete previous state survives.
8. Update the capability matrix, tutorial generator and inventory, migration
   notes, and release notes together. Run Ruff, the configured branch-coverage
   suite, relevant real solver/TESPy gates, warning-strict Sphinx, notebook
   checks, and installed-artifact smoke tests.

Existing nonthermal example
---------------------------

``target.heat_recovery_dt_min`` demonstrates the path without extending the
thermal serializer. Its application service performs an isolated inverse
calculation, returns ``contracts.heat_recovery_dt_min.HeatRecoveryDtMinResult``
with immutable provenance, and uses the specialist reporting adapter. The
result retains inverse-specific feasibility and thermodynamic-limit evidence;
it does not acquire ``Qh`` or other unrelated thermal report fields.

Aggregation contracts
---------------------

``contracts/report_metrics.py`` validates the policy on every report field.
The existing thermal payload weights operating metrics, takes the maximum of
HPR capital requirements, retains consensus metadata, and derives totals and
utility summaries. Period-specific HPR simulation records are excluded from a
weighted row and remain on their originating period rows. An aggregate is not
a simulated operating point.

Run ``scripts/generate_analysis_catalog.py`` and
``scripts/generate_tutorial_coverage.py`` after changing capabilities. Contract
tests compare the catalog with the public accessors, workspace mirrors, adapters,
and generated documentation.
