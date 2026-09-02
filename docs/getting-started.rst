Getting Started
===============

OpenPinch has two package-root workflow classes. Use :class:`PinchProblem
<OpenPinch.PinchProblem>` for one engineering case and :class:`PinchWorkspace
<OpenPinch.PinchWorkspace>` for named cases and scenarios.

Install the package, then run a complete heat-integration study:

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json", project_name="Site")
   problem.validate()
   problem.target.all_heat_integration()

   summary = problem.summary_frame()
   metrics = problem.metrics()
   report = problem.report()
   figure = problem.plot.grand_composite_curve()

The named target call is the only analysis step in this example. Validation,
summary, metrics, report, and plot operations inspect prepared or cached state;
they never decide which analysis to run.

Observation operations such as ``summary_frame()``, ``metrics()``, ``report()``,
and ``problem.plot.*`` are deliberately separate from execution. If the input,
configuration, or component inventory changes, run the desired target or design
method again before observing the refreshed result.

Focused Analysis
----------------

Use descriptive methods when only one analysis is required:

.. code-block:: python

   direct = problem.target.direct_heat_integration()
   total_site = problem.target.total_site_heat_integration()
   area_cost = problem.target.heat_exchanger_area_and_cost()

``all_heat_integration()`` is the dependency-aware convenience method. It
cycles through the zone hierarchy, computes direct targets where needed, and
then completes direct and utility-mediated indirect targeting across the Zone
hierarchy.

Specified Heat Recovery
-----------------------

When heat recovery is the requirement and global delta Tmin is unknown, use
the non-mutating inverse target:

.. code-block:: python

   recovery_approach = problem.target.heat_recovery_approach_temperature(
       heat_recovery={"value": 4_000.0, "unit": "kW"},
       period_id="0",
   )

   recovery_approach.approach_temperature
   recovery_approach.thermodynamic_limit
   recovery_approach.status

This returns process-level HRAT evidence without replacing the ordinary cached
target. See :doc:`guides/heat-recovery-approach-temperature` for supported
zones, multiperiod and workspace calls, validation, units, and the distinction
from exchanger-level EMAT.

Named Scenarios
---------------

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace("crude_preheat_train.json", project_name="Site")
   workspace.case("baseline").target.direct_heat_integration()

   tight = workspace.scenario("tight", dt_cont_multiplier=0.75)
   tight.target.direct_heat_integration()

   comparison = workspace.compare_cases("baseline", "tight")

Method arguments are one-call overrides. ``update_options(...)`` changes the
stored fallback for later calls. Configuration supplies engineering values; it
does not select which core method runs.

Next Steps
----------

- :doc:`guides/first-solve-python` for the full lifecycle.
- :doc:`guides/heat-recovery-approach-temperature` when recovery is specified.
- :doc:`api/pinchproblem` and :doc:`api/pinchworkspace` for interaction maps.
- :doc:`examples/notebook-series` for the nineteen maintained tutorials.
- :doc:`examples/tutorial-coverage-map` for the complete operation coverage.
