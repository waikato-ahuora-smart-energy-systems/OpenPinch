Zonal and Total Site Workflows
==============================

.. warning::

   This advanced guide uses unsupported internal application and analysis
   owners. Only :func:`OpenPinch.main.pinch_analysis_service` is compatibility
   protected.

Purpose
-------

Use zonal workflows when a study needs more than one process boundary. Zones
let OpenPinch solve local direct targets and then aggregate solved subzones
into Total Process or Total Site utility-system views.

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
   total_site = case.target.indirect_heat_integration()
   summary = case.summary_frame()

   print(summary[["Target", "Zone", "Hot Utility Target", "Cold Utility Target"]])

Expected Output
---------------

The summary contains target rows for different scopes and target families.
Direct rows describe local recovery inside a zone. Indirect or Total Site rows
describe utility-mediated recovery across solved subzones.

Interpretation
--------------

Compare zonal results by scope before comparing numbers:

- Check the target family: Direct Integration, Total Process, or Total Site.
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
