Zonal and Total Site Workflows
==============================

Purpose
-------

Use zonal workflows when a study needs more than one process boundary. Zones
let OpenPinch solve local direct targets and then aggregate solved subzones
into indirect utility-system views. An indirect result at Site scope is the
conventional Total Site target.

Prerequisites
-------------

You should understand the first-solve workflow and the difference between
direct and indirect integration. See :doc:`first-solve-python` and
:doc:`../fundamentals/direct-vs-indirect-integration`.

Sample Case
-----------

Use ``zonal_site.json`` for a compact multizone example or ``pulp_mill.json``
for a richer Total Site and cogeneration context.

Runnable Workflow
-----------------

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace(source="pulp_mill.json", project_name="Site")
   case = workspace.case("baseline")

   direct = case.target.direct_heat_integration()
   total_site = case.target.total_site_heat_integration()
   summary = case.summary_frame()

   print(
       summary[
           [
               "Scope",
               "Zone Type",
               "Integration Type",
               "Target Method",
               "Hot Utility Target",
               "Cold Utility Target",
           ]
       ]
   )

Expected Output
---------------

The summary contains target rows classified by canonical scope, Zone type,
integration route, and target method. Process / Heat Exchange rows describe
local recovery. Utility / Heat Exchange rows describe indirect recovery across
solved subzones.

Interpretation
--------------

Compare zonal results by scope before comparing numbers:

- Check Integration Type (Process or Utility) and Target Method.
- Check the zone name and hierarchy level.
- Compare hot and cold utility targets first.
- Use Grand Composite Curves, Total Site profiles, and SUGCC views to explain
  why the utility targets changed.

For multiperiod inputs, pass ``period_id=...`` to the targeting accessor:

.. code-block:: python

   winter_site = case.target.indirect_heat_integration(period_id="winter")

Next Steps
----------

- :doc:`../fundamentals/zones-streams-utilities-and-targets` for the model.
- :doc:`graphing-and-interpretation` for graph reading order.
- :doc:`notebooks-and-sample-cases` for notebook 02 and zonal sample cases.
