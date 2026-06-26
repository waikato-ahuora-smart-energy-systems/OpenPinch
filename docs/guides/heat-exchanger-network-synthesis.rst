Heat Exchanger Network Synthesis
================================

Purpose
-------

Use heat exchanger network synthesis when a solved OpenPinch case should be
converted into ranked network candidates and grid-diagram views.

Prerequisites
-------------

Install the synthesis extra and IDAES solver extensions before running
solver-backed HEN synthesis:

.. code-block:: bash

   python -m pip install "openpinch[synthesis]"
   idaes get-extensions

Source OpenHENS CSV files are migration source material only. Convert them
once into OpenPinch JSON or native ``TargetInput`` models before synthesis.

Sample Case
-----------

Use ``Four-stream-Yee-and-Grossmann-1990-1.json`` for the compact converted
OpenHENS benchmark used in the packaged HEN notebook.

Runnable Workflow
-----------------

.. code-block:: python

   from OpenPinch import PinchProblem
   from OpenPinch.lib import HENDesignMethod

   problem = PinchProblem(
       "Four-stream-Yee-and-Grossmann-1990-1.json",
       project_name="Four-stream converted OpenHENS example",
   )

   design = problem.design.enhanced_synthesis_method(quality_tier=2)
   network = design.network
   diagram = network.build_grid_diagram()

Explicit design-method accessors are also available:

.. code-block:: python

   problem.design.open_hens_method()
   problem.design.heat_exchanger_network_synthesis()
   problem.design.heat_exchanger_network_synthesis(
       method=HENDesignMethod.OpenHENS,
   )
   problem.design.pinch_design_method()
   problem.design.thermal_derivative_method(initial_networks=(seed_network,))
   problem.design.network_evolution_method(initial_networks=(existing_network,))

Expected Output
---------------

Successful synthesis stores a design result on ``TargetOutput.design`` and
``problem.results.design``. Inspect:

- ``design.design_method`` for the requested public design service
- ``design.manifest.method_sequence`` for executed task-level methods
- ``design.network`` for the selected network
- ``design.ranked_networks`` for ranked unique candidates
- ``design.network.build_grid_diagram(...)`` for visual topology inspection

Interpretation
--------------

The public design accessor is problem-rooted. Persistent synthesis controls
belong in loaded ``TargetInput.options`` keys such as ``HENS_APPROACH_TEMPERATURES``,
``HENS_METHOD_SEQUENCE``, ``HENS_SYNTHESIS_QUALITY_TIER``, solver names,
tolerance, output formats, and run id. Do not pass persistent design-space or
solver controls as a separate runtime object to the design call.

Use ``enhanced_synthesis_method(quality_tier=...)`` for the recommended
quality-tier workflow, ``open_hens_method()`` for the original tier 1 OpenHENS
sequence, and ``network_evolution_method(initial_networks=...)`` for retrofit
evolution from an existing network.

When Couenne is unavailable for Couenne-backed stages, OpenPinch warns and
attempts the configured network-evolution route where possible.

Migration and Support Notes
---------------------------

Old import paths and OpenHENS field aliases have been removed from the runtime
API. OpenPinch does not provide runtime import aliases, OpenHENS field aliases,
or command parity with the original OpenHENS scripts. Use the conversion
scripts and converted JSON case inputs instead.
The old import paths should be treated as migration-only references, not as
compatibility aliases.

For development checks, use:

.. code-block:: bash

   pytest -m "not synthesis and not solver"
   pytest -m synthesis
   pytest -m solver

Next Steps
----------

- :doc:`../examples/notebook-series` for notebook 09.
- :doc:`../api/schemas-and-config` for ``TargetOutput.design`` schemas.
- :doc:`../api/service-layer` for the internal method-oriented service stack.
