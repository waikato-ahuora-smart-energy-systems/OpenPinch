Decision Workflows
==================

This page maps common user questions to the right OpenPinch entry point,
example asset, and supporting documentation page.

Which Workflow Should I Use?
----------------------------

``Can I run a known-good case and inspect the main targets?``
   Use ``basic_pinch.json`` with :doc:`../guides/first-solve-cli` or
   :doc:`../guides/first-solve-python`.

``How sensitive is the answer to my minimum approach assumptions?``
   Use ``crude_preheat_train.json`` and
   ``01_basic_pinch_and_dtcont_sensitivity.ipynb``.

``How do multiple process areas aggregate into a site view?``
   Use ``zonal_site.json`` or ``pulp_mill.json`` together with
   :doc:`../guides/zonal-and-total-site-workflows` and
   ``02_total_site_targets_and_sugcc.ipynb``.

``Would an integrated heat pump improve the utility picture of my plant?``
   Use ``heat_pump_targeting.json`` with
   :doc:`../guides/heat-pump-workflows`. The dedicated
   explicit ``problem.target.direct_heat_pump(...)`` /
   ``problem.target.indirect_heat_pump(...)`` workflows
   route should currently be treated as experimental/partial.

``How do I compare direct and indirect HPR or refrigeration targets?``
   Use ``chocolate_factory.json`` and
   ``03_carnot_hpr_comparison.ipynb``.

``I need a typed request/response service contract, not a notebook wrapper.``
   Start from :doc:`../api/service-layer` and
   :doc:`../api/schemas-and-config`.

``I need to inspect prepared streams, zones, or problem tables directly.``
   Start from :doc:`../api/domain-model`.

Interpretation Sequence
-----------------------

Regardless of workflow, the recommended decision sequence is:

1. compare the hot and cold utility targets first
2. compare heat recovery and pinch temperatures second
3. inspect the most relevant graph family third
4. only then move into advanced scenario or equipment interpretation

That order keeps the package grounded in thermodynamic decision support rather
than graph-first exploration.
