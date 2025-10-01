API Reference
=============

High-level Services
-------------------

.. automodule:: OpenPinch.main
   :members: pinch_analysis_service, get_targets, get_visualise
   :show-inheritance:

Workflow Helpers
----------------

.. autoclass:: OpenPinch.classes.pinch_problem.PinchProblem
   :members:
   :show-inheritance:

Domain Models
-------------

.. autoclass:: OpenPinch.classes.stream.Stream
   :members:
   :show-inheritance:

.. autoclass:: OpenPinch.classes.zone.Zone
   :members:
   :show-inheritance:

Analysis Building Blocks
------------------------

.. automodule:: OpenPinch.analysis.data_preparation
   :members: prepare_problem_struture
   :show-inheritance:

.. automodule:: OpenPinch.analysis.process_analysis
   :members: get_process_pinch_targets
   :show-inheritance:

.. automodule:: OpenPinch.analysis.utility_targeting
   :members: get_zonal_utility_targets, target_utility, calc_GGC_utility
   :show-inheritance:
