Zonal and Total Site Workflows
==============================

OpenPinch becomes substantially more powerful once you move beyond a single
process-zone view and start using explicit zone hierarchies.

Question This Guide Answers
---------------------------

How do I use OpenPinch for nested process areas and higher-level utility
integration studies?

Why Zones Matter
----------------

Zones control analysis scope.

At small scale, a zone may represent:

- a unit operation
- a process area

At larger scale, zones can be aggregated into:

- a plant
- a site

Direct integration usually answers the local question. Indirect integration
usually answers the higher-level utility system question.

Practical Workflow
------------------

1. define a zone hierarchy explicitly when the study is multiscale
2. solve the local or process-zone picture
3. compare it to the aggregated Total Process or Total Site picture
4. use higher-level graph views to understand what changed

Useful Assets
-------------

Packaged sample:

.. code-block:: bash

   openpinch sample --name zonal_site.json -o zonal_site.json

Packaged notebook:

.. code-block:: bash

   openpinch notebook --name 02_total_site_targets_and_sugcc.ipynb -o notebooks

Python Pattern
--------------

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace(source="pulp_mill.json", project_name="Site")
   baseline = workspace.case("baseline")
   summary = baseline.summary_frame()
   total_site = baseline.target.indirect_heat_integration()

What To Compare
---------------

When comparing local and aggregated answers:

- compare hot utility targets
- compare cold utility targets
- compare graph types at the same scope
- confirm that target row names refer to the zone and target family you think
  they do

Useful Graphs
-------------

For multiscale workflows, the most useful views are often:

- grand composite curves
- Total Site profiles
- site utility grand composite curves

Next Steps
----------

- For the technical basis, see :doc:`../fundamentals/direct-vs-indirect-integration`.
- For graphs, see :doc:`graphing-and-interpretation`.
- For packaged assets, see :doc:`notebooks-and-sample-cases`.
