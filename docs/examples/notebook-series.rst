Notebook Series
===============

The packaged series teaches the public :class:`OpenPinch.PinchProblem` and
:class:`OpenPinch.PinchWorkspace` experience. Code cells use package-root
imports, contain no stored outputs, and declare one honest execution profile.

Core and Intermediate
---------------------

1. ``01_first_solve_and_core_curves.ipynb`` -- first solve, cached results,
   reports, and core curves (``base``).
2. ``02_focused_direct_and_total_site.ipynb`` -- focused direct, indirect, and
   Total Site analysis (``base``).
3. ``03_multisegment_streams.ipynb`` -- piecewise stream input and prepared
   segment inspection (``base``).
4. ``04_workspace_cases_and_scenarios.ipynb`` -- cases, scenarios, batches,
   configuration fallback, and comparison (``base``).
5. ``05_workspace_persistence.ipynb`` -- load, validation, case data, and
   bundle persistence (``base``).
6. ``06_multiperiod_heat_integration.ipynb`` -- ordered period targeting and
   weighted summaries (``base``).
7. ``07_area_cost_and_exergy.ipynb`` -- area/cost and exergy enrichment
   (``base``).
8. ``08_carnot_heat_pump_and_refrigeration.ipynb`` -- Carnot HPR placement and
   topology (``slow-hpr``).
9. ``09_vapour_compression_and_brayton.ipynb`` -- simulated vapour-compression
   and Brayton models (``slow-hpr``).
10. ``10_multiperiod_heat_pumps.ipynb`` -- mirrored multiperiod HPR methods
    (``slow-hpr``).
11. ``11_process_mvr_and_cascade.ipynb`` -- process MVR components and cascade
    HPR (``slow-hpr``).
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

Copy the Series
---------------

.. code-block:: bash

   openpinch notebook -o notebooks

Copy one notebook:

.. code-block:: bash

   openpinch notebook --name 01_first_solve_and_core_curves.ipynb -o notebooks

Profiles
--------

``base``
   Executes in routine CI from a clean temporary directory.

``slow-hpr``
   Requires the HPR model dependencies and a longer numerical run.

``solver``
   Requires the HEN synthesis extras and an available solver.

``interactive``
   Includes explicit filesystem or dashboard side effects and is run only in a
   guarded interactive environment.

The complete operation mapping and current profile policy are published in
:doc:`tutorial-coverage-map`.
