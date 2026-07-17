Pre-Release Notes
=================

Unreleased
----------

0.5.0 architecture clean break
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``OpenPinch.main.pinch_analysis_service`` is the sole current
  compatibility-protected Python contract. Its signature, validation order,
  return structure, serialization, ordering, exceptions, and numerical
  behaviour are unchanged.
- The package root is now an import-free marker. It no longer exports workflow
  classes, schemas, enums, resource helpers, or the main service.
- Business state, wire contracts, reusable optimisation, orchestration,
  engineering analysis, infrastructure adapters, and presentation now have
  explicit ``domain``, ``contracts``, ``optimisation``, ``application``,
  ``analysis``, ``adapters``, and ``presentation`` owners.
- ``OpenPinch.classes``, ``OpenPinch.lib``, ``OpenPinch.services``,
  ``OpenPinch.utils``, and ``OpenPinch.streamlit_webviewer`` are removed.
  There are no aliases, forwarding facades, dynamic export barrels, pickle
  shims, or migration package. Old deep imports and Python pickles are not
  migrated.
- Runtime stream segments, exchanger period states, exchanger area slices,
  process-MVR records, multiperiod HPR cases, dashboard state, graph build
  records, and HEN solver state remain private to their parent or service
  owner.
- Optimisation is a reusable package-level capability. Heat-pump targeting
  crosses one explicit adapter; new services can reuse optimisation without
  importing heat-pump code.
- HPR internal records are attribute-only, and HPR optimiser identifiers must
  exactly match ``dual_annealing``, ``cmaes``, ``bo``, or ``rbf_surrogate``.
  Obsolete helper-signature retries and historical optimiser spellings are not
  accepted.
- ``StreamCollection`` supports current-version pickle round trips, including
  deterministic fallback for an unpicklable callable sort key. It does not
  repair state dictionaries produced by older package versions.
- Internal type-only imports now resolve to the concrete configuration owner,
  total-site utility profiles forward the canonical period keyword, and invalid
  crossflow row counts raise an explicit validation error.
- Tests now mirror observable owner layers. The external main contract is
  authoritative, architecture directions are AST-enforced, and CI measures
  seeded statement and branch coverage with Hypothesis seed ``20260715``.

This is an intentional pre-1.0 clean break. No data migration, import migration,
pickle migration, or compatibility layer is provided.

Domain and input contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~

This pre-release changes the following contracts without compatibility shims:

- Values owned by a ``Stream`` and its segment records are read-only. Mutations use
  explicit stream assignment, indexed-value, or segment-update APIs.
- Period weights use one validation policy: omitted trailing weights become
  ``1.0``; excess, non-finite, negative, and all-zero vectors are rejected.
- Structured process-stream and nested thermal-profile inputs reject unknown
  fields. Process streams accept the canonical ``name`` and
  ``heat_capacity_flowrate`` spellings only.
- Workspace bundles use schema version ``2`` and ``case_input``. Version ``1``,
  unknown versions, and the retired ``payload`` field are rejected.
- Segmented process streams and utilities share the same semantic validation in
  reports and preparation, including parent aggregate consistency.

Private helper ownership
~~~~~~~~~~~~~~~~~~~~~~~~

- Private class helpers are grouped under owner-oriented packages for streams,
  values, collections, problem tables, problem orchestration, workspaces, and
  heat exchangers.
- Runtime stream-segment, exchanger-period, and exchanger-area-slice record
  classes are parent-owned implementation details. Construct them through
  ``Stream`` and ``HeatExchanger`` mappings. The equivalent segment wire shape
  remains accepted through ``pinch_analysis_service(...)``; direct schema
  imports are not compatibility-protected.
- The former public record imports and their Python pickle paths are removed
  without aliases or compatibility shims.
- Synthesis schemas now have concrete common, topology, method, task, and result
  owners. The compatibility-only ``methods``, ``tasks``, and ``results`` modules
  and the synthesis package re-export barrel are removed. Import concrete owner
  modules instead; old barrel-qualified pickle paths are unsupported.
- The package root and owner package ``__init__`` files are import-free markers.
  The retired ``classes``, ``lib``, ``services``, ``utils``, and
  ``streamlit_webviewer`` package paths have no compatibility facades; advanced
  callers import concrete owner modules.
- Process-MVR records, multiperiod HPR period cases, dashboard graph state,
  graph specifications/metadata, and HEN solver runtime records are private to
  their owning services. Concrete parent components, schemas, direct-MVR
  models, and HEN equation-model classes remain available as unsupported
  internals.

Period-native PDM and utility constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- PDM decompositions expose ordered ``period_targets`` with explicit period
  identities and indices. Clipped temperatures and active-stream flags are
  period-indexed; the retired singular target and clipped-temperature fields do
  not exist.
- Each operating period uses its own utility targets and pinch temperature.
  Shared topology is the union of streams and matches active in any period, and
  a pinch side is solved when any period requires it.
- Above- and below-pinch amalgamation retains every period's duties,
  temperatures, approach variables, split fractions, and explicit
  non-isothermal branch outlet temperatures.
- Non-isothermal warm starts normalize hot split fractions across cold matches
  and cold split fractions across hot matches independently in every period.
- Segmented utilities use local per-segment ``dt_cont`` values. The inlet uses
  the first segment contribution, the solved match outlet uses the traversed
  segment contribution, and an exact boundary uses the larger adjacent value.

Period-native HEN results
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``HeatExchanger`` retains shared topology, design area, and capital fields.
  Operational duty, activity, approaches, split fractions, and source/sink
  temperatures are stored in non-empty ordered ``period_states`` containing
  parent-owned period-state records.
- The retired exchanger-level operating scalar fields do not exist. Use
  ``exchanger.state(period_id)``; omission is accepted only for an exchanger
  with exactly one period state.
- Multiperiod duty, temperature, diagram, export, and controllability queries
  require ``period_id``. No implicit period-zero selection is provided.
- Extraction walks every period array, retains matches active only outside the
  first period, and prefers explicit non-isothermal branch outlet temperatures.
  Solved branch split fractions keep downstream duty checks physically valid.

Isolated summaries and HPR economics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Multiperiod summary replay captures one baseline zone and solves every period
  against a fresh deep copy. The original zone object, cached results, and
  recorded targeting specification are restored after success or failure.
- Shared simulated-HPR candidates are ranked by weighted operating cost plus
  weighted feasibility penalty plus maximum annualized capital cost. Weighted
  backend ``obj`` is used only for backends that provide no cost breakdown.
- Weighted internal HPR summaries average operating fields, use the maximum
  capital, annualized-capital, compressor-capital, and heat-exchanger-capital
  fields, and recompute total annualized cost as weighted operating cost plus
  maximum annualized capital. Non-HPR fields retain weighted averaging.
