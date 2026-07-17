Support and Stability
=====================

OpenPinch 0.5.0 intentionally has one compatibility-protected Python contract.
This narrow promise lets the internal package architecture improve without
maintaining forwarding modules or import aliases.

Stable
------

The supported Python call is:

.. code-block:: python

   from OpenPinch.main import pinch_analysis_service

Its signature, validation order, exceptions, return shape, serialization,
ordering, and numerical behaviour are protected by the end-to-end contract
suite. The request and response wire structures used by this function are also
protected. The root ``OpenPinch`` package is an import-free marker and exports
no user objects.

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
