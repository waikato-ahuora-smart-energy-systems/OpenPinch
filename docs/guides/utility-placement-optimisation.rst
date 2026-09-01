Utility Placement Optimisation
==============================

Purpose
-------

Utility placement optimisation selects the temperatures of multiple hot and
cold candidate utility levels while preserving the process heat targets. For
each candidate, the ordinary hierarchy-aware target workflow determines the
period-specific duties. The service minimizes physical entropy generation
calculated from the resulting balanced hot and cold composite curves and
reports the result in ``kW/K``.

The workflow is available through the Python API only; no CLI surface is
provided. Monetary utility-placement optimisation is deferred for now.
General cogeneration analyses elsewhere in OpenPinch are unchanged.

Prerequisites
-------------

Load a valid :class:`OpenPinch.PinchProblem` directly or select one from a
:class:`OpenPinch.PinchWorkspace`. When counts are supplied, ``isothermal``
must be at least 2 and creates that many levels on each utility side;
``sensible`` optionally adds sensible levels on each side.

Count-generated levels are temperature-coupled by kind and ordinal. For
example, ``hot_iso_1`` and ``cold_iso_1`` occupy the same interval with
opposite direction: the cold supply equals the hot target, and the cold target
equals the hot supply. The same rule applies to generated sensible pairs.
Their temperatures are shared, but ordinary targeting determines each side's
duty independently.
Generated isothermal and sensible pairs may interleave, and their physical
sequence is determined from optimized supply temperatures. Existing Hot and
Cold utilities inferred when counts are omitted retain independent temperature
coordinates.

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
the existing templates with generated levels. ``HU`` and ``CU`` are reserved
balancing fallbacks and are never inferred as placement options.

Utility placement uses ``cmaes`` as its default black-box optimizer. Select a
different supported backend per call with ``options={"method": ...}``; the
exact alternatives are ``dual_annealing``, ``bo``, and ``rbf_surrogate``.
Backend selection is independent of the HPR ``HPR_BB_MINIMISER`` setting.

Runnable Workflow
-----------------

Run Process-level placement against the ``Almond`` direct GCC with four
isothermal and zero sensible levels on each side:

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace(
       source="chocolate_factory.json", project_name="Site"
   )
   problem = workspace.use_case("baseline")

   process_case = problem.target.utility_placement(
       isothermal=4,
       sensible=0,
       zone="Almond",
       period_ids=("0",),
   )
   process_evidence = process_case.utility_placement_result

Notebook 19 uses this uncapped form. The optional ``maximum_duties`` API remains
available outside that tutorial and is keyed by the final globally unique
utility name. Omitted names are unbounded and zero disables only that named
level. A scalar applies to every selected period. Explicit units and period
identities are also accepted:

.. code-block:: python

   maximum_duties = {
       "hot_iso_1": {"value": 0.02, "unit": "MW"},
       "cold_iso_1": {
           "values": [10.0, 15.0],
           "period_ids": ["summer", "winter"],
           "unit": "kW",
       },
   }

Each hot and cold level has its own independent bound, including matching
generated pairs. Explicit identities are resolved against the problem's
canonical period order. If only a subset is selected, the returned case leaves
that utility unbounded in unselected periods. When named capacity cannot cover
the target, residual-only
``HU`` or ``CU`` supplies the shortfall and is retained in the returned case.
Its supply temperature is fixed 50 K above the context-wide maximum process
temperature for ``HU`` or 50 K below the context-wide minimum for ``CU``. This
keeps fallback last in ordinary utility targeting.
The optimizer changes temperatures and sensible spans only. For every
candidate and period, OpenPinch installs those utilities in a detached case and
runs the same direct, Total Site, or indirect target method used after the
optimization. That exact target determines named duties, same-level Total Site
utility matching, target totals, and fallback duty. A requested candidate level
may therefore have zero duty; there is no implicit minimum-duty constraint.
Retargeting replaces prior utility duties, including writing exact zero for
capped, unused, and zero-load levels.

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
       isothermal=4,
       sensible=0,
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
plot method for a deterministic graph mapping. Retargeted named and fallback
duties match the optimizer evidence because both use the same ordinary target
workflow.

For a Process target, ordinary utility targeting limits each sensible duty at
the tightest endpoint or interior GCC breakpoint. The cumulative hot and cold
utility profiles therefore remain on the feasible side of the residual Process
GCC over their complete temperature spans; a sensible Utility GCC must not
cross the Process GCC. This same check applies during every detached optimizer
replay and when the returned case is targeted again.

Generated temperature bounds cover the intervals where each residual profile
changes, rather than forcing every hot level above the hottest process
temperature or every cold level below the coldest. Deterministic starting
points distribute utility supplies and sensible spans across that support, so
even a deliberately small tutorial search can compare utilities near different
parts of the background profile. These starts improve coverage but do not turn
the bounded search into a proof of the global optimum.

Internally, all optimizer-facing coordinates are normalized to ``[0, 1]``.
Generated isothermal and sensible levels may interleave; targeting orders them
by their optimized supply temperatures. Temperature decoding retains stable
same-kind identities, and independent verification enforces the minimum gap
between physically adjacent supplies. By default that gap is
``0.01 delta_degC``, the same as the default isothermal temperature difference.
Returned utility temperatures remain ordinary physical values.

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

Each returned object is a normal case with the best hot and cold candidate
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
``HU`` and ``CU`` are not placement options. They are residual-only balancing
utilities positioned 50 K beyond the process-temperature extremes; positive
fallback duty remains feasible. Its separate
dimensionless penalty in period ``p`` is

.. math::

   g_p = 1000 \left[\left(\frac{Q_{HU,p}}{Q_{heat,required,p}}\right)^2
       + \left(\frac{Q_{CU,p}}{Q_{cool,required,p}}\right)^2\right].

For ranking, OpenPinch maps the weighted physical entropy monotonically using
an entropy-valued reference derived from the process slices, then combines it
with ``sum(w_p * g_p)``. The optimizer backend, canonical replay, alternatives,
and selected best result all use this same scalar ranking. The reported
physical entropy remains the balanced-composite result in ``kW/K``; the
dimensionless penalty is exposed separately as ``fallback_penalty``.

Temperatures and duties retain explicit units. Default canonical units are
``degC``, ``delta_degC``, ``kW``, and ``kW/K``. Invalid input is rejected before
optimization, including a level count below "at least 2", duplicate names,
infeasible cross-period bounds, or exhaustion without a feasible candidate.

Next Steps
----------

Run ``19_utility_placement_optimisation.ipynb`` for the executable
thermodynamic workflow, four isothermal and zero sensible levels per side,
named-case replacement, inspectable optimizer-versus-retarget duty comparison
tables, and standard GCC and Total Site Profile plots. Replace
the sample with reviewed site data and defensible bounds, then increase
``iteration_limit`` and ``evaluation_limit`` beyond the tutorial's deliberately
small values before using the result for engineering decisions.
