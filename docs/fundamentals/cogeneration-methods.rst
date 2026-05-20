Cogeneration Methods
====================

OpenPinch includes turbine cogeneration analysis as a post-processing workflow
on top of the thermal targeting results.

Above Pinch Logic
-----------------

Above Pinch cogeneration interprets suitable hot utility demands as candidate
steam extraction levels in a multistage turbine train. The workflow estimates:

- turbine work target
- turbine efficiency target
- stage-level extraction details

This is a thermodynamic screening and targeting workflow, not a full plant
equipment design package.

Below Pinch Logic
-----------------

Below Pinch cogeneration interprets suitable thermal loads against an
environmental sink as a condensing turbine opportunity. It answers a different
question from the above Pinch extraction path, even though the same turbine
solver family is reused.

Important Configuration Ideas
-----------------------------

The main turbine inputs are grouped on `zone.config`, including:

- `TURB_T_IN`
- `TURB_P_IN`
- `MIN_EFF`
- `LOAD_FRACTION`
- `ETA_MECH`
- `TURB_MODEL`
- `IS_HIGH_P_COND_FLASH`

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
4. use the stage details only after the high-level answer looks promising

Recommended Follow-On Pages
---------------------------

- :doc:`../guides/cogeneration-workflows`
- :doc:`../api/pinchproblem`
- :doc:`../api/generated-index`
