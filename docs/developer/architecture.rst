Package Architecture
====================

This page documents the internal architecture used to implement the sole
external call, :func:`OpenPinch.main.pinch_analysis_service`. Internal modules
are not compatibility-protected, but their ownership and dependency direction
are enforced by tests.

Owner Responsibilities
----------------------

.. list-table::
   :header-rows: 1

   * - Owner
     - Responsibility
     - May depend on
   * - ``domain``
     - Business state, invariants, arithmetic, indexing, parent-owned records
     - domain peers and third-party numerical/value libraries
   * - ``contracts``
     - Request, response, report, workspace, HPR, and HEN wire models
     - contracts and domain
   * - ``optimisation``
     - Reusable scalar optimisation models, candidates, execution, and backends
     - optimisation peers, NumPy, and SciPy
   * - ``adapters``
     - Filesystem formats, optional-dependency checks, and infrastructure translation
     - adapters, contracts, domain, and resource lookup
   * - ``analysis``
     - Deterministic engineering calculations, HPR, HEN, and numerical services
     - analysis, contracts, domain, optimisation, and explicit adapter leaves
   * - ``application``
     - Use-case orchestration, caches, replay, workspaces, and coordination
     - inward owners and lazy presentation calls
   * - ``presentation``
     - Reports, tables, figures, dashboards, diagrams, and exports
     - presentation, analysis, adapters, contracts, and domain

Dependency Direction
--------------------

Dependencies point toward domain and contracts. Domain never imports an outer
owner. Contracts do not import application, analysis, adapters, optimisation,
or presentation. Optimisation is reusable without importing heat-pump code.
Application coordinates concrete owners but does not import solver backends,
Plotly, Streamlit, or filesystem classes directly. Optional dependencies are
loaded only by the leaf that owns the feature.

The architecture test records a small exact set of existing cross-layer HEN
service and type dependencies. The test fails if those exceptions spread to a
new file. This makes remaining coupling visible rather than silently treating
it as a general dependency direction.

Execution Flow
--------------

The supported request follows this sequence:

1. ``OpenPinch.main`` validates the external request through ``contracts``.
2. ``application`` constructs and coordinates the problem use case.
3. ``analysis`` performs deterministic targeting using ``domain`` state.
4. ``presentation`` and ``adapters`` shape outputs only when requested by an
   internal advanced workflow.
5. ``OpenPinch.main`` returns the validated output contract.

Optimisation Boundary
---------------------

``OpenPinch.optimisation`` accepts an immutable scalar problem, method, and
options and returns ranked candidates. Heat-pump analysis translates its
thermodynamic objective at one explicit adapter. Other services can reuse the
same optimiser without importing any HPR module. No mutable backend registry or
service locator is used.

Private Runtime Records
-----------------------

Stream segments, exchanger period states, exchanger area slices, process-MVR
records, multiperiod cases, graph build specifications, dashboard state, and
solver records belong to their parent or service owner. They may be inspected
through parent results, but their classes are not external construction APIs.

Test Architecture
-----------------

Tests mirror observable owners under ``tests/e2e``, ``tests/application``,
``tests/domain``, ``tests/analysis``, ``tests/optimisation``,
``tests/adapters``, ``tests/presentation``, and ``tests/contracts``.
Architecture and packaging tests are separate. Private-module tests are kept
for mathematical kernels, solver equation order, explicit architecture rules,
or focused failure localization.

The end-to-end main suite is the authoritative compatibility suite. Property
tests use Hypothesis with seed ``20260715`` while retaining shrinking.
