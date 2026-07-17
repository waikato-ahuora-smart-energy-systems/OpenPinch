Support and Stability
=====================

Supported Process-Engineer Surface
----------------------------------

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace

The two package-root classes and their public target, all-period target,
component, design-result, and plot operations form the supported workflow
surface. Their exact released inventory is checked against the canonical
:doc:`../examples/tutorial-coverage-map`.

Stability Levels
----------------

``Stable``
   Package-root case management, direct and Total Site heat integration,
   reporting, and the documented observation lifecycle.

``Advanced``
   Heat Pump, cogeneration, exergy, process MVR, multiperiod, and HEN design
   methods exposed through descriptive ``problem.target`` and
   ``problem.design`` callables.

``Experimental / partial``
   Individual solver backends and contributor-level analysis modules whose
   availability or completeness is stated in their method guide.

Optional feature profiles are installed with ``openpinch[notebook]`` for the
tutorial environment, ``openpinch[dashboard]`` for Streamlit presentation, and
``openpinch[synthesis]`` for solver-backed HEN design.

Contributor Modules
-------------------

Concrete analysis, domain, optimisation, adapter, and presentation modules are
implementation owners. They remain documented for contributors but are not
alternative process-engineer entry points.

Clean-Break Policy
------------------

Retired stateful workflow spellings are removed without aliases. Configuration
does not select core methods. Public changes must update the live inventory,
tutorial manifest, executable notebooks, API pages, and release notes together.
