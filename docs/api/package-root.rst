External Python Contract
========================

OpenPinch exposes the two high-level workflow coordinators from the package
root:

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace

The strict mapping-in/result-out service remains available separately:

.. code-block:: python

   from OpenPinch.main import pinch_analysis_service

The package root ``__all__`` contains exactly ``PinchProblem`` and
``PinchWorkspace``. Schemas, enums, resources, lower-level services, and
``pinch_analysis_service`` stay with their concrete owners.

Workflow Classes
----------------

Use ``PinchProblem`` for a single live study and ``PinchWorkspace`` for named
baseline and variant cases. Their implementation owners remain
``OpenPinch.application.problem`` and ``OpenPinch.application.workspace``;
application code should use the shorter root imports shown above.

Service Function
----------------

.. autofunction:: OpenPinch.main.pinch_analysis_service
   :no-index:

Contract
--------

The protected signature is:

.. code-block:: python

   pinch_analysis_service(data, project_name="Project")

``data`` is a caller mapping accepted by the request contract. The call returns
the structured target-output model. Validation errors, field ordering,
serialization, and numerical results are part of the same protected boundary.

Example
-------

.. code-block:: python

   from OpenPinch.main import pinch_analysis_service

   result = pinch_analysis_service(
       {
           "streams": [
               {
                   "name": "Hot feed",
                   "zone": "Process",
                   "t_supply": 180.0,
                   "t_target": 80.0,
                   "heat_flow": 1000.0,
               },
               {
                   "name": "Cold feed",
                   "zone": "Process",
                   "t_supply": 20.0,
                   "t_target": 120.0,
                   "heat_flow": 800.0,
               },
           ],
           "utilities": [],
       },
       project_name="example",
   )

   print(result.model_dump(mode="json"))

Internal Owners
---------------

Concrete modules under ``application``, ``analysis``, ``domain``,
``contracts``, ``adapters``, ``optimisation``, and ``presentation`` are
documented for contributors and advanced experiments. Apart from the two
selected root workflow exports, they are not external compatibility contracts.
No other deep import, package barrel, or Python pickle path is preserved across
releases.

See :doc:`../developer/architecture` for the internal dependency map and
:doc:`../overview/support-and-stability` for the support policy.
