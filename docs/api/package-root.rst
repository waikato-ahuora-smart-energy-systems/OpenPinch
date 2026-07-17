External Python Contract
========================

OpenPinch currently compatibility-protects exactly one Python import:

.. code-block:: python

   from OpenPinch.main import pinch_analysis_service

The package root is an import-free marker. It has no ``__all__`` declaration
and does not re-export workflow classes, schemas, enums, resources, or the main
service.

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

Unsupported Internal Owners
---------------------------

Concrete modules under ``application``, ``analysis``, ``domain``,
``contracts``, ``adapters``, ``optimisation``, and ``presentation`` are
documented for contributors and advanced experiments. They are not external
compatibility contracts. No deep import, root alias, package barrel, or Python
pickle path is preserved across releases.

See :doc:`../developer/architecture` for the internal dependency map and
:doc:`../overview/support-and-stability` for the support policy.
