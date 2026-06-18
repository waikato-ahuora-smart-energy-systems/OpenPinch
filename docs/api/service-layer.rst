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

Stateful inputs remain state-aware after preparation, but state selection does
not happen inside ``prepare_problem(...)``. Instead, the selected state is
applied later through the targeting-service ``args`` dictionaries or the higher
level ``problem.target.*(..., state_id=...)`` wrappers.

.. autofunction:: OpenPinch.services.input_data_processing.data_preparation.prepare_problem
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

   input_data = TargetInput.model_validate(payload)
   zone = data_preprocessing_service(input_data, project_name="Example")

   direct_heat_integration_service(zone, {"state_id": "peak"})
   indirect_heat_integration_service(zone, {"state_id": "peak"})

Each targeting service mutates the prepared zone in place, records the
requested state metadata on the zone, and adds or refreshes the corresponding
target model.

The exergy service follows a slightly different contract from the base thermal
targeting services: it enriches an already existing compatible target for the
requested state instead of re-solving direct or indirect targeting internally.

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
