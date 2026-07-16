Service Layer
=============

The service layer is the boundary between validated input data and the
prepared/solved in-memory model. It is the right integration surface when you
want more control than :class:`~OpenPinch.classes.pinch_problem.PinchProblem`
provides but do not want to invoke individual low-level algorithms directly.

Layering
--------

The service stack is designed in three steps:

1. validate or receive typed request data
2. prepare the inputs into a :class:`~OpenPinch.classes.zone.Zone` hierarchy
3. dispatch direct, indirect, HPR, exergy, cogeneration, or area/cost targeting

Use Cases
---------

Use the service layer when you need to:

- embed OpenPinch in another application with a stable request/response
  boundary
- prepare a zone hierarchy once and run multiple advanced studies against it
- inspect the prepared model before solving
- mutate a live prepared model with process components before rerunning targets
- bypass file handling entirely and work with typed inputs
- apply exergy or cogeneration as post-processing on already solved targets

Main Service Surface
--------------------

.. automodule:: OpenPinch.services
   :members:
   :no-index:

Preparation Entry Point
-----------------------

The preparation stage is the key boundary between external inputs and the
internal model. It validates configuration choices, builds the zone tree,
applies ``dt_cont`` multipliers, instantiates process and utility streams, and
produces the ``Zone`` object consumed by the solver stack.

Period-valued inputs remain period-aware after preparation, but period selection does
not happen inside ``prepare_problem(...)``. Instead, the selected period is
applied later through the targeting-service ``args`` dictionaries or the higher
level ``problem.target.*(..., period_id=...)`` wrappers.

.. autofunction:: OpenPinch.services.input_data_processing.data_preparation.prepare_problem
   :no-index:

Heat Exchanger Network Synthesis Entry
--------------------------------------

Heat exchanger network synthesis is problem-rooted. User code should normally
enter through ``PinchProblem.design``:

.. code-block:: python

   from OpenPinch.lib import HENDesignMethod

   problem.design.enhanced_synthesis_method(quality_tier=2)
   problem.design.open_hens_method()
   problem.design.heat_exchanger_network_synthesis()
   problem.design.heat_exchanger_network_synthesis(
       method=HENDesignMethod.NetworkEvolution,
       initial_networks=(existing_network,),
   )

The internal service entry point owns method dispatch and final result caching.
It dispatches to the same direct services exposed by the design accessor.
Use ``enhanced_synthesis_method(quality_tier=...)`` as the public quality-tier
selector, ``open_hens_method()`` for original tier 1 OpenHENS, and
``heat_exchanger_network_synthesis()`` for the generic fast tier 0 default or
explicit enum dispatch.

.. automodule:: OpenPinch.services.heat_exchanger_network_synthesis.heat_exchanger_network_synthesis_entry
   :members:
   :no-index:

The HEN synthesis package is intentionally method-oriented:

- ``targeting_services`` contains method-specific orchestration.
- ``common.execution`` contains settings, task builders, executor contracts,
  and fallback policy.
- ``common.results`` contains result assembly and seed lookup.
- ``common.reporting`` contains ranking, verification, and export helpers.
- ``common.solver`` contains optional-dependency checks, array adapters,
  backend calls, and network extraction.
- ``unit_models`` contains the equation/unit model layer for pinch design and
  stagewise models.

Network Grid Diagrams
---------------------

The selected heat exchanger network can construct its own grid diagram through
:meth:`OpenPinch.classes.heat_exchanger_network.HeatExchangerNetwork.build_grid_diagram`:

.. code-block:: python

   design = problem.results.design
   period_id = design.network.period_ids[0]
   diagram = design.network.build_grid_diagram(period_id=period_id)

The standalone service remains available for batch rendering one or more
:class:`~OpenPinch.classes.heat_exchanger_network.HeatExchangerNetwork`
objects, for example when displaying several ranked candidates.

.. automodule:: OpenPinch.services.network_grid_diagram
   :members:
   :no-index:

Network Controllability
-----------------------

Solved heat exchanger networks can also be screened for steady-state
controllability. The service treats process-stream outlet temperatures as
controlled outputs and practical bypass or utility-flow adjustments as
manipulated variables, then scores the resulting duty-normalised interaction
matrix.

For multiperiod networks, pass ``period_id`` to both diagram and
controllability services. OpenPinch does not silently select period zero.

.. code-block:: python

   design = problem.results.design
   period_id = design.network.period_ids[0]
   assessment = design.network.quantify_controllability(period_id=period_id)
   assessment.score
   assessment.components.rank

The score is a screening metric rather than a dynamic closed-loop simulation.
It is intended for comparing candidate HEN topologies and identifying networks
with weak actuator coverage, poor pairing, low thermal margin, or insufficient
control redundancy.

.. automodule:: OpenPinch.services.heat_exchanger_network_controllability
   :members:
   :no-index:

Typical Preparation and Solve Pattern
-------------------------------------

.. code-block:: python

   from OpenPinch.lib.schemas.io import TargetInput
   from OpenPinch.services import (
       data_preprocessing_service,
       direct_heat_integration_service,
       indirect_heat_integration_service,
   )

   source_data = {"streams": [...], "utilities": [...]}
   input_data = TargetInput.model_validate(source_data)
   zone = data_preprocessing_service(input_data, project_name="Example")

   direct_heat_integration_service(zone, {"period_id": "peak"})
   indirect_heat_integration_service(zone, {"period_id": "peak"})

Each targeting service mutates the prepared zone in place, records the
requested period metadata on the zone, and adds or refreshes the corresponding
target model.

The exergy service follows a slightly different contract from the base thermal
targeting services: it enriches an already existing compatible target for the
requested period instead of re-solving direct or indirect targeting internally.

Direct High-Level Orchestration
-------------------------------

For callers that want one function rather than an object wrapper, the root
orchestration helper remains available:

.. automodule:: OpenPinch.main
   :members:
   :no-index:

Choosing Between Interfaces
---------------------------

- Use :func:`OpenPinch.main.pinch_analysis_service` when you want a typed
  request/response contract.
- Use ``problem.add_component.process_mvr(...)`` when the study needs direct
  gas/vapour MVR stream replacement before ordinary target reruns.
- Use :mod:`OpenPinch.services` when you want to prepare a zone once and run
  several advanced analyses against the same prepared state.
- Use :class:`~OpenPinch.classes.pinch_problem.PinchProblem` when you want a
  notebook- and file-oriented convenience wrapper with summaries and exports.

The service layer is also the best place to look when you are trying to
understand how the narrative workflow maps onto the actual analysis pipeline.
