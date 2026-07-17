Package-Root API
================

Process-engineer workflows begin with exactly these imports:

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace

``PinchProblem`` owns one prepared case. ``PinchWorkspace`` owns named cases,
scenario creation, case batches, comparison, and persistence. Analysis methods
are grouped under descriptive accessors such as ``problem.target``,
``problem.design``, ``problem.components``, and ``problem.plot``.

The package root intentionally remains small. Schemas, domain records, and
analysis services have concrete owner modules for contributors, but process
engineers do not need those imports for supported workflows.

.. autoclass:: OpenPinch.PinchProblem
   :no-index:

.. autoclass:: OpenPinch.PinchWorkspace
   :no-index:

See Also
--------

- :doc:`pinchproblem`
- :doc:`pinchworkspace`
- :doc:`../examples/tutorial-coverage-map`
