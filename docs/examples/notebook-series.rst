Notebook Series
===============

The packaged series teaches the public :class:`OpenPinch.PinchProblem` and
:class:`OpenPinch.PinchWorkspace` experience. Code cells use package-root
imports, contain no stored outputs, and declare one honest execution profile.

Every notebook follows the same lifecycle: prepare the study, run one named
engineering method, then inspect the cached results. Observation cells read
results only and never launch analysis. Arguments on a method call apply to
that analysis; stored configuration is only the fallback when an argument is
omitted. The sample data ships with OpenPinch, so the notebooks run without path
setup; read the named inputs and assumptions before substituting plant data.

Core and Intermediate
---------------------

1. ``01_first_solve_and_core_curves.ipynb`` -- first solve, cached results,
   reports, and core curves (``base``).
2. ``02_focused_direct_and_total_site.ipynb`` -- focused direct, inverse
   threshold-problem global ``dt_min``, indirect, and Total Site analysis, including
   the positive thermodynamic-limit boundary, frozen result fields, and
   non-mutation check (``base``).
3. ``03_multisegment_streams.ipynb`` -- piecewise stream input and prepared
   segment inspection (``base``).
4. ``04_workspace_cases_and_scenarios.ipynb`` -- cases, scenarios, batches,
   configuration fallback, and comparison (``base``).
5. ``05_workspace_persistence.ipynb`` -- load, validation, case data, and
   bundle persistence (``base``).
6. ``06_multiperiod_heat_integration.ipynb`` -- ordered period targeting,
   scalar and mapped equivalent global ``dt_min`` requests, and weighted summaries
   (``base``).
7. ``07_area_cost_and_exergy.ipynb`` -- area/cost and exergy enrichment
   (``base``).
8. ``08_carnot_heat_pump_and_refrigeration.ipynb`` -- bounded one-stage Carnot
   heat-pump and refrigeration targets, plots, and residual utility placement
   (``slow-hpr``).
9. ``09_vapour_compression_and_brayton.ipynb`` -- comprehensive simulated
   vapour-compression targeting, detached winning records, target-owned performance maps,
   plain-data export, required CoolProp refrigeration, explicit optional TESPy
   mixtures, and optional Brayton comparisons (``slow-hpr``).
10. ``10_multiperiod_heat_pumps.ipynb`` -- five separate bounded shared-design
    optimizations across ``turndown``, ``base``, and ``peak``, plus one optional
    advanced cascade screen (``slow-hpr``).
11. ``11_process_mvr_and_cascade.ipynb`` -- direct process-MVR stage/work
    evidence and a required bounded one-stage CoolProp VC+MVR target
    (``slow-hpr``).
12. ``12_cogeneration.ipynb`` -- default and named turbine models (``base``).
13. ``13_multiperiod_cogeneration.ipynb`` -- multiperiod cogeneration
    (``base``).
14. ``14_energy_transfer.ipynb`` -- site energy-transfer analysis and diagrams
    (``base``).

HEN Design and Publication
--------------------------

15. ``15_hen_synthesis_and_selection.ipynb`` -- ranked HEN synthesis,
    selection, utilities, serialization, and grids (``solver``).
16. ``16_advanced_hen_methods.ipynb`` -- enhanced, OpenHENS, Pinch Design,
    thermal-derivative, and evolution methods (``solver``).
17. ``17_multiperiod_hen_synthesis.ipynb`` -- one shared multiperiod HEN
    (``solver``).
18. ``18_results_plots_reports_exports.ipynb`` -- complete observation and
    explicit export/dashboard surfaces (``interactive``).
19. ``19_utility_placement_optimisation.ipynb`` -- entropy-based placement of
    four temperature-coupled isothermal hot/cold utility levels (no sensible
    levels) at Process and Site scope, replacement in a new case, and standard
    GCC and Total Site Profile plots (``base``).

Copy the Series
---------------

.. code-block:: bash

   openpinch notebook -o notebooks

Copy one notebook:

.. code-block:: bash

   openpinch notebook --name 01_first_solve_and_core_curves.ipynb -o notebooks

Profiles
--------

Every notebook is executed by CI from a clean temporary directory. The
profile decides which CI lane runs it:

``base``
   Runs in the routine unit lane.

``slow-hpr``
   Requires the HPR model dependencies (``tespy`` extra) and a longer
   numerical run; executed in the dedicated ``notebooks-hpr`` lane.

``solver``
   Requires the HEN synthesis extras and an available solver; executed in the
   solver job on changes bound for ``main``.

``interactive``
   Includes explicit filesystem or dashboard side effects. It runs in the unit
   lane with the dashboard launch guarded while real plots and workbook
   exports execute.

The complete operation mapping and current profile policy are published in
:doc:`tutorial-coverage-map`.
