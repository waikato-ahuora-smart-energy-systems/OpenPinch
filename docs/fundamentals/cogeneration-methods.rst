Cogeneration Methods
====================

OpenPinch includes turbine cogeneration analysis as a post-processing workflow
on top of the thermal targeting results.

Above Pinch Logic
-----------------

Above Pinch cogeneration interprets suitable hot utility demands as candidate
steam extraction levels in a multistage turbine train. By default it uses the
hot utilities of the zone's Indirect (Total Site) target when one exists, then
the HPR targets, then the Direct target; pass ``base_target=`` to choose one.
A utility at or above water's critical temperature, outside water saturation,
above the turbine inlet pressure, or with no duty is not a turbine stage. The
workflow estimates:

- turbine work target
- turbine efficiency target
- stage-level extraction details, as ``turbine_result``

This is a thermodynamic screening and targeting workflow, not a full plant
equipment design package.

Below Pinch Logic
-----------------

Below Pinch cogeneration would interpret suitable thermal loads against an
environmental sink as a condensing turbine opportunity. It answers a different
question from the above Pinch extraction path, even though the same turbine
solver family is reused. The solver function exists, but the public
``problem.target.*cogeneration`` methods do not call it; they run the above
Pinch path only.

Important Configuration Ideas
-----------------------------

The main turbine inputs are the option keys below. They are read at runtime
from ``zone.config.power`` (for example ``zone.config.power.turb_t_in``):

- ``POWER_TURB_T_IN``
- ``POWER_TURB_P_IN``
- ``POWER_MIN_EFF``
- ``POWER_LOAD_FRACTION``
- ``POWER_ETA_MECH``
- ``POWER_HIGH_P_COND_FLASH_ENABLED``

The turbine model is selected by the target method name, not by an option:

- ``cogeneration``, the default, uses Medina-Flores et al. (2010)
- ``sun_smith_cogeneration`` uses Sun and Smith (2015)
- ``varbanov_small_turbine_cogeneration`` and
  ``varbanov_large_turbine_cogeneration`` use the two size bands of Varbanov et
  al. (2004): stage work at or below 2000 kW, and above 2000 kW
- ``isentropic_cogeneration(efficiency=...)`` uses a fixed isentropic
  efficiency, greater than 0 and at most 1, in place of ``POWER_MIN_EFF``

Using a Varbanov method outside its band gives a warning; see
:doc:`../guides/cogeneration-workflows`.

This keeps the turbine workflow aligned with the rest of the package model:
configuration lives on the prepared zone, not in workbook-style side channels.

Interpretation Guidance
-----------------------

Use cogeneration outputs as supporting decision context after the thermal
targets are understood.

Recommended order:

1. understand the base thermal target
2. confirm the utility levels are meaningful for turbine screening
3. inspect cogeneration work and efficiency targets

Recommended Follow-On Pages
---------------------------

- :doc:`../guides/cogeneration-workflows`
- :doc:`../api/pinchproblem`
- :doc:`../api/generated-index`
