Heat Pump and Refrigeration
===========================

This package is an unsupported advanced owner. Its concrete imports and parent
accessors may change without a compatibility layer.

The Heat Pump and refrigeration stack is the most specialised part of the
OpenPinch codebase. It combines preprocessing of background cascades,
thermodynamic cycle models, and the package-level scalar optimisation service
to screen direct and indirect integration opportunities.

Where To Start
--------------

Repository developers can begin with the parent-owned internal surfaces
documented in :doc:`api-core`:

- ``problem.target.direct_heat_pump(...)``
- ``problem.target.indirect_heat_pump(...)``
- ``problem.target.direct_refrigeration(...)``
- ``problem.target.indirect_refrigeration(...)``

The modules on this page are the lower-level implementation layers behind
those helpers.

Package Overview
----------------

.. automodule:: OpenPinch.analysis.heat_pumps
   :no-members:
   :no-index:

Public HPR Entrypoints
----------------------

.. automodule:: OpenPinch.analysis.heat_pumps.service
   :members:

Shared Preprocessing and Optimisation Boundary
----------------------------------------------

The targeting parsers decode optimiser vectors into temperatures, ambient
duties, base duty scales, split vectors, and process availability arrays. The
aggregate backend classes then allocate requested duties from base/split
coordinates, clip those requests to availability, and add any excess to the
penalty term. Leaf physical unit models receive only concrete model duties.
Simulated vapour-compression backends then combine the HPR streams with the
background and ambient streams into one residual GCC. The pocket-free GCC end
points provide residual external utilities for operating-cost accounting;
cycle penalties remain separate feasibility terms. HPR objective and failure
semantics are translated to the reusable optimiser only by
``optimisation_adapter``; the generic optimiser has no heat-pump dependency.
Optimiser identifiers are exact: ``dual_annealing``, ``cmaes``, ``bo``, and
``rbf_surrogate``. Case changes, surrounding whitespace, and abbreviated or
historical spellings are rejected.

.. automodule:: OpenPinch.analysis.heat_pumps.common
   :no-members:

.. automodule:: OpenPinch.analysis.heat_pumps.common.encoding
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.common.preprocessing
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.common.shared
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.optimisation_adapter
   :members:

HPR Schemas
-----------

``HPRParsedState`` and ``HPRBackendResult`` are internal typed records with
attribute-only access. Use their named attributes while processing results and
``model_dump()`` when a mapping is required; they do not emulate dictionaries.

.. autoclass:: OpenPinch.contracts.hpr.HPRParsedState
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr.HeatPumpTargetInputs
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr.HPRBackendResult
   :members:
   :no-index:

.. autoclass:: OpenPinch.contracts.hpr.SimulatedHPRAnnualizedCostAccounting
   :members:
   :no-index:

Cycle Optimisation Services
---------------------------

These modules place or size Heat Pump and refrigeration cycle models against
prepared cascade data. The detailed cycle physics live in the concrete
``cycles`` modules documented in :doc:`api-classes`.

Only the current internal cycle names are routed here, for example
``"Cascade Carnot cycles"``, ``"Parallel Carnot cycles"``, and
``"Parallel vapour compression cycles"``.

.. automodule:: OpenPinch.analysis.heat_pumps.targeting
   :no-members:

.. automodule:: OpenPinch.analysis.heat_pumps.targeting.brayton
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.targeting.parallel_carnot
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.targeting.parallel_vapour_compression
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.targeting.cascade_carnot
   :members:

.. automodule:: OpenPinch.analysis.heat_pumps.targeting.vapour_compression_mvr
   :members:
