Notebook Series
===============

OpenPinch ships with a packaged notebook series that is treated as part of the
supported learning and regression surface. Each notebook is built around one
decision question rather than a generic feature tour. The series now spans the
main single-case API, named multi-case studies, multistate targeting, and the
typed/service boundary.

Included Notebooks
------------------

``01_basic_pinch_and_dtcont_sensitivity.ipynb``
   Single-case ``PinchProblem`` front door, direct graph interpretation,
   ``area_cost()`` appendix, and ``PinchWorkspace``-based ``dt_cont``
   sensitivity on the real crude preheat train case.

``02_total_site_targets_and_sugcc.ipynb``
   Packaged ``pulp_mill.json`` Total Site workflow, including scope-ladder
   comparison, a real ``Bleaching`` local GCC screen, serialized graph output
   inspection, and local-versus-site cogeneration screens.

``03_carnot_hpr_comparison.ipynb``
   Advanced HPR and refrigeration comparison on ``chocolate_factory.json``
   across the direct heat pump, indirect heat pump, direct refrigeration, and
   indirect refrigeration workflows.

``04_multistate_targeting_and_state_comparison.ipynb``
   Real named-state targeting on ``crude_preheat_train_multistate.json`` and
   ``zonal_site_multistate.json``, including ``state_ids``,
   ``target_all_states()``, and cross-state utility tables.

``05_schema_service_and_output_workflows.ipynb``
   ``copy_sample_case(...)``, local-file ``PinchProblem``, typed
   ``TargetInput`` plus ``pinch_analysis_service(...)``, artifact export, and
   serialized ``PinchWorkspace`` variant views.

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

1. Start with the basic pinch and ``dt_cont`` notebook to understand the
   single-case workflow and the named-study sensitivity pattern.
2. Move to the Total Site notebook once you are comfortable with zonal and
   indirect integration concepts.
3. Use the Carnot HPR notebook after you understand how to read the base
   utility and graph outputs.
4. Move to the multistate notebook when your case has named operating states or
   seasonal variation.
5. Use the schema/service notebook when you need typed validation, serialized
   workspace views, or repeatable export workflows.

Why These Matter
----------------

The notebooks do more than demonstrate commands. They reveal the practical
power of the package: direct single-case solves, named-case comparison,
hierarchical targeting, graph-based interpretation, real multistate studies,
and stable programmatic boundaries built on the same packaged assets. The
distributed copies are also kept output-free so they do not ship stale plots,
tracebacks, or machine-specific execution state.
