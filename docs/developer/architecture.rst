Package Architecture
====================

This page documents the internal architecture used to implement
:class:`OpenPinch.PinchProblem` and :class:`OpenPinch.PinchWorkspace`. Module
ownership and dependency direction are enforced by
``tests/architecture/test_dependency_rules.py``.

Owner Responsibilities
----------------------

.. list-table::
   :header-rows: 1

   * - Owner
     - Responsibility
     - May depend on
   * - ``domain``
     - Business state, invariants, arithmetic, indexing, parent-owned records
     - domain peers, pydantic, CoolProp (loaded lazily), and numerical/value libraries
   * - ``contracts``
     - Request, response, report, workspace, heat pump and refrigeration (HPR), and heat exchanger network (HEN) wire models
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
     - inward owners and presentation; optional presentation leaves load lazily
   * - ``presentation``
     - Reports, tables, figures, dashboards, diagrams, and exports
     - presentation, analysis, adapters, contracts, and domain; never
       application or optimisation

Dependency Direction
--------------------

Dependencies point toward domain and contracts. Domain never imports an outer
owner. Contracts do not import application, analysis, adapters, optimisation,
or presentation. Optimisation is reusable without importing heat-pump code.
Application coordinates concrete owners but does not import solver backends,
Plotly, Streamlit, or filesystem classes directly. Optional dependencies are
loaded only by the leaf that owns the feature.

Report adapters live in ``OpenPinch/presentation/reporting/`` and follow the
presentation row above: they may import ``analysis``, ``adapters``,
``contracts`` and ``domain``, but not ``application`` or ``optimisation``.
The ``application`` owner calls them to serialise results.

The architecture test (``test_layer_dependencies_follow_explicit_allowed_directions``) records no cross-layer exceptions: the set of allowed
exceptions is empty, and the test fails if any new cross-layer import appears.

Execution Flow
--------------

The supported request follows this sequence:

1. ``PinchProblem`` validates input through ``contracts``.
2. ``application`` constructs and coordinates the problem use case.
3. ``analysis`` performs deterministic targeting using ``domain`` state.
4. ``application`` serialises results through ``presentation`` on every target
   run; ``adapters`` translate input formats, and reports, exports and plots
   are produced on request.
5. ``PinchProblem`` caches the validated output contract for observation.

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
