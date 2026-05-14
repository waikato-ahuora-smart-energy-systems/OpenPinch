Notebook Series
===============

OpenPinch ships with a packaged notebook series that is treated as part of the
supported learning and regression surface. Each notebook is built around one
decision question rather than a generic feature tour.

Included Notebooks
------------------

``01_basic_pinch_and_dtcont_sensitivity.ipynb``
   Baseline direct-integration workflow, summary reading, graph inspection, and
   ``dt_cont`` sensitivity using the public ``PinchProblem`` API.

``02_total_site_targets_and_sugcc.ipynb``
   Zonal and total-site workflow on a pulp-mill style case, including indirect
   targeting and site utility grand composite interpretation.

``03_carnot_hpr_comparison.ipynb``
   Direct and indirect Carnot HPR and refrigeration comparison using the
   advanced ``problem.target.*`` entry points.

How To Use Them
---------------

Copy the full series:

.. code-block:: bash

   openpinch notebook -o notebooks

Copy one notebook:

.. code-block:: bash

   openpinch notebook --name 02_total_site_targets_and_sugcc.ipynb -o notebooks

Recommended Learning Order
--------------------------

1. Start with the basic pinch and ``dt_cont`` notebook to understand the main
   process-level workflow and output interpretation.
2. Move to the total-site notebook once you are comfortable with zonal and
   indirect integration concepts.
3. Use the Carnot HPR notebook after you understand how to read the base
   utility and graph outputs.

Why These Matter
----------------

The notebooks do more than demonstrate commands. They reveal the practical
power of the package: scenario comparison, hierarchical targeting, graph-based
interpretation, and advanced thermal integration studies on top of the same
prepared data model.
