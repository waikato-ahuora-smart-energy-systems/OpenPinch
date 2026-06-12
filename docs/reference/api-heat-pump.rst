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

.. automodule:: OpenPinch.services.heat_pump_integration.common
   :no-members:

.. automodule:: OpenPinch.services.heat_pump_integration.common.encoding
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.common.preprocessing
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.common.shared
   :members:

Cycle Optimisation Services
---------------------------

These modules place or size Heat Pump and refrigeration cycle models against
prepared cascade data. The detailed cycle physics live in the
``unit_models`` modules documented in :doc:`api-classes`.

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services
   :no-members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.brayton
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.cascade_vapour_compression
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.multi_simple_carnot
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.multi_simple_vapour_compression
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.multi_temperature_carnot
   :members:

.. automodule:: OpenPinch.services.heat_pump_integration.targeting_services.vapour_compression_mvr
   :members:
