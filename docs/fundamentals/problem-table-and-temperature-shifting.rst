Problem Tables and Temperature Shifting
=======================================

The numerical core of OpenPinch is the Problem Table style interval cascade.
This is the bridge between raw stream temperatures and the reported utility and
pinch targets.

Why Temperature Shifting Exists
-------------------------------

Real heat exchange requires a finite temperature approach. OpenPinch models
that requirement by shifting stream temperatures before building the cascade.

In the runtime stream model:

- hot streams shift downward using the active approach value
- cold streams shift upward using the active approach value

This is why the package distinguishes between:

- original physical temperatures
- shifted temperatures used for targeting calculations

The shifted view answers the practical recovery question, not just the ideal
overlap question.

OpenPinch Runtime Representation
--------------------------------

At the class level, the stream model stores:

- the base ``delta_t_contribution``
- the ``effective_delta_t_contribution``
- shifted temperatures such as `t_min_star` and `t_max_star`

That allows workflows such as sensitivity studies or zone-level multiplier
updates without overwriting the original input assumption.

What The Problem Table Produces
-------------------------------

Once streams are shifted and grouped into temperature intervals, the cascade
can be used to derive:

- hot utility target
- cold utility target
- heat recovery
- pinch temperatures
- graph-ready curve data

The same broad machinery supports both direct process-level targeting and the
site-style aggregation workflows, although the stream sets being cascaded are
different.

Real and Shifted Outputs
------------------------

OpenPinch exposes both thermodynamic numbers and graph interpretations. This
is why you will see multiple graph families:

- composite curves
- shifted composite curves
- balanced composite curves
- grand composite curves

The shifted views are usually the best place to connect the numerical targets
back to a practical integration picture.

Implications For Sensitivity Work
---------------------------------

Any workflow that changes effective approach assumptions should be expected to
change:

- shifted temperatures
- interval boundaries
- utility targets
- graph shapes

This is why OpenPinch rebuilds prepared runtime structures when some
configuration-level assumptions change, rather than pretending those values are
harmless scalar tweaks.

Recommended Follow-On Pages
---------------------------

- :doc:`pinch-analysis`
- :doc:`zones-streams-utilities-and-targets`
- :doc:`graphs-and-interpretation`
