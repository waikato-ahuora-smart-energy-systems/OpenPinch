Heat Pump and Refrigeration
===========================

The Heat Pump and refrigeration stack is the most specialised part of the
OpenPinch codebase. It combines preprocessing of background cascades,
thermodynamic cycle models, and black-box optimisation to screen direct and
indirect integration opportunities.

Where To Start
--------------

Most users should still begin with the higher-level surfaces documented in
:doc:`api-core`:

- ``problem.target.direct_heat_pump(...)``
- ``problem.target.indirect_heat_pump(...)``
- ``problem.target.direct_refrigeration(...)``
- ``problem.target.indirect_refrigeration(...)``

The modules on this page are the lower-level implementation layers behind
those helpers.

Package Overview
----------------

.. automodule:: OpenPinch.services.heat_pump_integration
   :no-members:

Public HPR Entrypoints
----------------------

.. automodule:: OpenPinch.services.heat_pump_integration.heat_pump_and_refrigeration_entry
   :members:

Shared Preprocessing and Optimisation Helpers
---------------------------------------------

The targeting parsers decode optimiser vectors into temperatures, ambient
duties, base duty scales, split vectors, and process availability arrays. The
aggregate backend classes then allocate requested duties from base/split
coordinates, clip those requests to availability, and add any excess to the
penalty term. Leaf physical unit models receive only concrete model duties.
Simulated vapour-compression backends then combine the HPR streams with the
background and ambient streams into one residual GCC. The pocket-free GCC end
points provide residual external utilities for operating-cost accounting;
cycle penalties remain separate feasibility terms.

.. automodule:: OpenPinch.services.heat_pump_integration.common
   :no-members:

.. automodule:: OpenPinch.services.heat_pump_integration.common.encoding
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.common.preprocessing
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.common.shared
   :members:

HPR Schemas
-----------

.. autoclass:: OpenPinch.lib.schemas.hpr.HeatPumpTargetInputs
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.hpr.HPRBackendResult
   :members:
   :no-index:

.. autoclass:: OpenPinch.lib.schemas.hpr.SimulatedHPRAnnualizedCostAccounting
   :members:
   :no-index:

Cycle Optimisation Services
---------------------------

These modules place or size Heat Pump and refrigeration cycle models against
prepared cascade data. The detailed cycle physics live in the
``unit_models`` modules documented in :doc:`api-classes`.

Only the current public cycle names are routed here, for example
``"Cascade Carnot cycles"``, ``"Parallel Carnot cycles"``, and
``"Parallel vapour compression cycles"``.

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services
   :no-members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.brayton
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.cascade_vapour_compression
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.parallel_carnot
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.parallel_vapour_compression
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.cascade_carnot
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.vapour_compression_mvr
   :members:
