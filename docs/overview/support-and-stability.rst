Support and Stability
=====================

OpenPinch intentionally keeps a narrow public Python surface so that the
internal package architecture can improve without broad forwarding modules or
import aliases.

Stable
------

The supported high-level workflow imports are:

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace

For strict mapping-in/result-out integrations, use:

.. code-block:: python

   from OpenPinch.main import pinch_analysis_service

The service signature, validation order, exceptions, return shape,
serialization, ordering, and numerical behaviour are protected by the
end-to-end contract suite. The request and response wire structures used by
this function are also protected. The package root exports exactly the two
workflow classes; other objects remain with concrete owner modules.

Advanced
--------

Concrete modules beneath ``OpenPinch.application``, ``OpenPinch.analysis``,
``OpenPinch.domain``, ``OpenPinch.contracts``, ``OpenPinch.adapters``,
``OpenPinch.optimisation``, and ``OpenPinch.presentation`` are maintained as a
coherent internal architecture. They are inspectable and tested, but their
Python import paths and signatures are not compatibility promises.

Experimental / partial
----------------------

Solver-backed HEN work, simulated heat-pump cycles, dashboards, repository
resource helpers, and packaged advanced notebooks may depend on optional
software or internal owner modules. Their numerical fixtures are regression
tested where dependencies are available, but callers should expect structural
changes before a future contract expansion is explicitly selected.

Dependency Boundaries
---------------------

Optional dependencies are workflow-specific:

- ``openpinch[notebook]`` for Jupyter, Plotly, and Excel tooling
- ``openpinch[dashboard]`` for Streamlit review
- ``openpinch[synthesis]`` plus solver extensions for HEN synthesis
- ``openpinch[brayton_cycle]`` for TESPy-backed cycles

Optional packages load only in their owner leaves and report an actionable
installation extra when absent.

No Migration Facades
--------------------

Version 0.5.0 provides no aliases, forwarding modules, dynamic export barrels,
or pickle-path shims for removed imports. In particular, the retired
``OpenPinch.classes``, ``OpenPinch.lib``, ``OpenPinch.services``,
``OpenPinch.utils``, and ``OpenPinch.streamlit_webviewer`` paths do not resolve.

Next Steps
----------

- :doc:`../api/package-root` for the exact external call.
- :doc:`../developer/architecture` for internal ownership and dependency rules.
- :doc:`capability-matrix` for workflow maturity and optional dependencies.
