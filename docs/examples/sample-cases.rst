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

``crude_preheat_train_multistate.json``
   Named-state derivative of the crude preheat train for real multistate direct
   targeting and cross-state comparison.

``zonal_site.json``
   Hierarchical site-style case for multi-zone and aggregated Total Site
   workflows.

``zonal_site_multistate.json``
   Seasonal named-state derivative of the site-style case for multistate Total
   Site and indirect integration workflows.

``pulp_mill.json``
   Larger site-style example used in the packaged Total Site notebook.

``heat_pump_targeting.json``
   Compact HPR screening input data for direct Heat Pump experimentation,
   including the current ``Parallel vapour compression cycles`` option,
   without the larger multi-case notebook flow.

``chocolate_factory.json``
   Advanced direct-versus-indirect Carnot HPR and refrigeration example used in
   the packaged notebook workflow.

``Four-stream-Yee-and-Grossmann-1990-1.json``
   Converted OpenHENS Ye and Grossman four-stream heat exchanger network
   synthesis example used by notebook 09.

How To Copy Them
----------------

Browse or copy the packaged cases through :mod:`OpenPinch.resources` when you
want a local editable copy:

.. code-block:: python

   from OpenPinch.resources import copy_sample_case, list_sample_cases

   print(list_sample_cases())
   copy_sample_case("basic_pinch.json", "basic_pinch.json")

You can also load a packaged case by name directly from Python when no local
file with that name exists:

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace

   problem = PinchProblem("basic_pinch.json")
   workspace = PinchWorkspace(
       source="crude_preheat_train.json",
       project_name="crude_preheat_train",
   )

Choosing The Right Case
-----------------------

- Use ``basic_pinch.json`` for first-run validation and graph reading.
- Use ``crude_preheat_train.json`` for process-only sensitivity studies.
- Use ``crude_preheat_train_multistate.json`` when the same process needs named
  operating states such as turndown, base, and peak.
- Use ``zonal_site.json`` or ``pulp_mill.json`` when you need a real site-style
  indirect integration example.
- Use ``zonal_site_multistate.json`` when the site answer needs seasonal or
  named-state comparison.
- Use ``heat_pump_targeting.json`` when you want a smaller direct HPR
  screening input data set.
- Use ``chocolate_factory.json`` when you want to study the advanced
  direct-versus-indirect HPR and refrigeration surface used by notebook 03.
- Use ``Four-stream-Yee-and-Grossmann-1990-1.json`` when you want the converted
  OpenHENS four-stream synthesis case used by notebook 09.
