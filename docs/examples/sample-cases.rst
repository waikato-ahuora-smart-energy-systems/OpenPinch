Sample Cases
============

The packaged sample cases are not filler assets. They are the fastest path to a
reproducible OpenPinch workflow and each one corresponds to a specific part of
the supported surface.

Included Cases
--------------

``basic_pinch.json``
   Minimal direct-integration process example for first-run validation and
   baseline graph interpretation.

``crude_preheat_train.json``
   Process-only case that is useful for direct integration and ``dt_cont``
   sensitivity studies.

``zonal_site.json``
   Hierarchical zone-tree case for multi-zone and aggregated total-site style
   workflows.

``pulp_mill.json``
   Larger site-style example used in the packaged total-site notebook.

``heat_pump_targeting.json``
   Case intended for the helper-backed integrated heat-pump comparison path and
   related HPR experimentation.

``chocolate_factory.json``
   Advanced direct-versus-indirect Carnot HPR and refrigeration example used in
   the packaged notebook workflow.

How To Copy Them
----------------

Copy one case:

.. code-block:: bash

   openpinch sample --name basic_pinch.json -o basic_pinch.json

List or browse the packaged cases through :mod:`OpenPinch.resources` if you are
driving the workflow from Python.

Choosing The Right Case
-----------------------

- Use ``basic_pinch.json`` for first-run validation and graph reading.
- Use ``crude_preheat_train.json`` for process-only sensitivity studies.
- Use ``zonal_site.json`` or ``pulp_mill.json`` when you need a real zone-tree
  and indirect integration example.
- Use ``heat_pump_targeting.json`` when you want the helper-backed HPR
  comparison workflow.
- Use ``chocolate_factory.json`` when you want to study the advanced
  ``problem.target`` HPR and refrigeration surface.
