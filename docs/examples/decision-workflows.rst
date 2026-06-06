Decision Workflows
==================

This page maps common user questions to the right OpenPinch entry point,
example asset, and supporting documentation page.

Which Workflow Should I Use?
----------------------------

``Can I run a known-good case and inspect the main targets?``
   Use ``basic_pinch.json`` with :doc:`../guides/first-solve-python`. If you
   want the packaged learning path first, use
   :doc:`../guides/first-solve-cli` to copy notebook 01, then solve inside
   Python from that notebook.

``How sensitive is the answer to my minimum approach assumptions?``
   Use ``crude_preheat_train.json`` and
   ``01_basic_pinch_and_dtcont_sensitivity.ipynb``. The notebook uses
   ``PinchWorkspace.copy_case(...)`` and ``set_dt_cont_multiplier(...)`` to
   keep the comparison explicit.

``How do multiple process areas aggregate into a site view?``
   Use ``zonal_site.json`` or ``pulp_mill.json`` together with
   :doc:`../guides/zonal-and-total-site-workflows` and
   ``02_total_site_targets_and_sugcc.ipynb``.

``How do named operating states change the answer across one process or site?``
   Use ``crude_preheat_train_multistate.json`` or
   ``zonal_site_multistate.json`` together with
   ``04_multistate_targeting_and_state_comparison.ipynb``.

``Would an integrated heat pump improve the utility picture of my plant?``
   Use ``heat_pump_targeting.json`` with
   :doc:`../guides/heat-pump-workflows`. The dedicated explicit
   ``problem.target.direct_heat_pump(...)`` /
   ``problem.target.indirect_heat_pump(...)`` workflows are the supported
   advanced route.

``How do I compare direct and indirect HPR or refrigeration targets?``
   Use ``chocolate_factory.json`` and
   ``03_carnot_hpr_comparison.ipynb``. That workflow stays on the public
   ``problem.target.*`` and ``problem.plot.*`` surfaces rather than lower-level
   cycle internals.

``I need a typed request/response service contract, not a notebook wrapper.``
   Start from :doc:`../api/service-layer`,
   :doc:`../api/schemas-and-config`, and
   ``05_schema_service_and_output_workflows.ipynb``.

``I need to inspect prepared streams, zones, or Problem Tables directly.``
   Start from :doc:`../api/domain-model`.

Interpretation Sequence
-----------------------

Regardless of workflow, the recommended decision sequence is:

1. compare the hot and cold utility targets first
2. compare heat recovery and pinch temperatures second
3. inspect the most relevant graph family third
4. only then move into advanced study-case or equipment interpretation

That order keeps the package grounded in thermodynamic decision support rather
than graph-first exploration.
