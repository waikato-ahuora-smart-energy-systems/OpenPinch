Utility Placement Optimisation
==============================

Purpose
-------

Utility placement optimisation selects the temperatures of multiple hot and
cold utility levels while preserving the process heat targets. It minimizes
physical entropy generation calculated from candidate-local balanced hot and
cold composite curves and reports the result in ``kW/K``.

The workflow is available through the Python API only; no CLI surface is
provided. Monetary utility-placement optimisation is deferred for now.
General cogeneration analyses elsewhere in OpenPinch are unchanged.

Prerequisites
-------------

Load a valid :class:`OpenPinch.PinchProblem` directly or select one from a
:class:`OpenPinch.PinchWorkspace`. When counts are supplied, ``isothermal``
must be at least 2 and creates that many levels on each utility side;
``sensible`` optionally adds sensible levels on each side.

The selected hierarchy node determines the target profile. With no ``zone``,
the optimizer uses the problem's master zone. A Process Zone or Unit Operation
uses its direct GCC, a Site uses its Total Site Profile, and a Community or
Region uses its indirect aggregate profile. Supply a unique zone name, full
zone address, or owned ``Zone`` object to select another node. There is no
public target-type string to coordinate separately.

If both counts are omitted, OpenPinch infers templates from the existing utilities.
A missing or near-equal target temperature is isothermal; a larger
span, segmented utility, or multi-point profile is sensible. ``Both`` utilities
form paired hot and cold templates. Supplying either count explicitly replaces
the existing templates with generated levels.

Runnable Workflow
-----------------

Run Process-level placement against the ``Almond`` direct GCC with two
isothermal and two sensible levels on each side:

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace(
       source="chocolate_factory.json", project_name="Site"
   )
   problem = workspace.use_case("baseline")

   process_case = problem.target.utility_placement(
       isothermal=2,
       sensible=2,
       zone="Almond",
       period_ids=("0",),
   )
   process_evidence = process_case.utility_placement_result

The return value is already a normal detached :class:`OpenPinch.PinchProblem`
containing the best utility set. Register it, then target and plot it with the
standard public workflow. The baseline remains unchanged:

.. code-block:: python

   process_case = workspace.add(
       process_case,
       name="optimized_process_utilities",
       activate=False,
   )

   process_case.target.direct_heat_integration(zone="Almond", period_id="0")
   process_summary = process_case.summary_frame()
   process_gcc = process_case.plot.grand_composite_curve(zone_name="Almond")

Run Site-level placement separately. Because the master zone is a Site, no
``zone`` argument is required:

.. code-block:: python

   site_case = problem.target.utility_placement(
       isothermal=2,
       sensible=2,
       period_ids=("0",),
   )
   site_case = workspace.add(
       site_case,
       name="optimized_site_utilities",
       activate=False,
   )
   site_case.target.total_site_heat_integration(period_id="0")
   site_summary = site_case.summary_frame()
   site_tsp = site_case.plot.total_site_profiles()

The standard GCC and Total Site Profile figures include the utilities stored
on their corresponding returned cases. Use ``return_graph_data=True`` on either
plot method for a deterministic graph mapping.

Use ``problem.target.all_periods.utility_placement(...)`` to select every
canonical period. An ordered case batch uses the same arguments:

.. code-block:: python

   outcome = workspace.cases(["baseline"]).target.utility_placement(
       isothermal=2,
       period_ids=("0",),
   )
   successful = outcome.results
   failures = outcome.errors

Expected Output
---------------

Each returned object is a normal unsolved case with the best hot and cold
utilities already present. Use ordinary case targeting, ``summary_frame()``,
``metrics()``, ``report()``, and plotting methods. Detailed optimizer evidence
remains available at ``process_case.utility_placement_result`` or
``site_case.utility_placement_result``. It contains
the selected periods, termination evidence, alternatives, diagnostics, entropy
generation, and exergy destruction, but normal use does not require traversing
that specialist structure.

Interpretation
--------------

OpenPinch adds the allocated candidate utilities to the real-temperature
process composites and forms balanced hot and cold composite curves. On every
common heat-load interval ``j``, sensible entropy generation is

.. math::

   \dot S_{gen} = \sum_j \left[
   \dot C_{P,c,j}\ln\!\left(\frac{T_{c,out,j}}{T_{c,in,j}}\right)
   + \dot C_{P,h,j}\ln\!\left(\frac{T_{h,out,j}}{T_{h,in,j}}\right)
   \right].

Temperatures in logarithms are absolute. An isothermal segment uses the signed
``Q/T`` limit. Lower non-negative entropy generation means less thermodynamic
irreversibility; moving a utility temperature closer to its matched process
temperature lowers the objective. Multiplying by ambient absolute temperature
gives exergy destruction.

For multiple selected periods, the objective is the raw weighted sum
``sum(w_p * S_gen,p)``. It is not divided by total weight, and a candidate must
be feasible in every selected period. Generated fallback utilities named
``HU`` and ``CU`` are not placement options: any positive duty assigned to
either receives a deterministic infeasible penalty.

Temperatures and duties retain explicit units. Default canonical units are
``degC``, ``delta_degC``, ``kW``, and ``kW/K``. Invalid input is rejected before
optimization, including a level count below "at least 2", duplicate names,
infeasible cross-period bounds, or exhaustion without a feasible candidate.

Next Steps
----------

Run ``19_utility_placement_optimisation.ipynb`` for the executable
thermodynamic workflow, two isothermal plus two sensible levels per side,
named-case replacement, and standard GCC and Total Site Profile plots. Replace
the sample with reviewed site data and defensible bounds, then increase
``iteration_limit`` and ``evaluation_limit`` beyond the tutorial's deliberately
small values before using the result for engineering decisions.
