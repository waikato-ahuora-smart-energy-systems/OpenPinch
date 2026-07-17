Core Application API
====================

The process-engineer API is documented in :doc:`../api/package-root`,
:doc:`../api/pinchproblem`, and :doc:`../api/pinchworkspace`.

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace

   problem = PinchProblem("basic_pinch.json", project_name="Site")
   problem.target.all_heat_integration()
   summary = problem.summary_frame()

Contributor-facing implementation modules are listed in the generated module
index. They are concrete owners, not additional application entry points.
