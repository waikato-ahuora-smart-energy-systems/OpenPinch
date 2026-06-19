Heat Exchanger Network Synthesis
================================

Heat exchanger network synthesis is exposed through the same problem and
workspace roots as the rest of OpenPinch. A heat exchanger network run starts from an
OpenPinch-compatible JSON payload, a native
:class:`~OpenPinch.lib.schemas.io.TargetInput`, or an already loaded
:class:`~OpenPinch.PinchProblem`. Source OpenHENS CSV files are migration
source material only; convert them once into OpenPinch JSON or native
``TargetInput`` payloads before running synthesis.

The implementation boundary is
``heat_exchanger_network_synthesis_service(problem)``. It is internal and
problem-rooted: it requires a live ``PinchProblem`` and reads persistent heat exchanger network
configuration from ``TargetInput.options`` through the prepared
``Configuration``. User code should call the problem design accessor or the
workspace workflow dispatch shown below, not the internal service directly.
Those public calls invoke the local solver-backed synthesis executor by
default.

When Couenne is unavailable for the Couenne-backed derivative/topology stages,
OpenPinch emits a warning and attempts ``energy_stage_refinement`` directly
with the configured ESM solver and stage selection.

Problem Workflow
----------------

The direct replacement for the OpenHENS README study example is a
``PinchProblem`` whose input options carry the synthesis grid, method sequence,
solver names, tolerance, output formats, and run id.

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem(
       source="Four-stream-Yee-and-Grossmann-1990-1.json",
       project_name="Four-stream converted OpenHENS example",
   )

   design = problem.design.heat_exchanger_network_synthesis()

   result = problem.results
   network = result.design.network

   for exchanger in network.exchangers:
       print(exchanger.source_stream, "->", exchanger.sink_stream, exchanger.duty)

The ``source`` file above is an OpenPinch-compatible JSON payload. The original
OpenHENS CSV example is not loaded at runtime by synthesis. Its process streams
map to ``StreamSchema`` records, utilities and utility prices map to
``UtilitySchema`` records, and the synthesis controls map to ``TargetInput``
``options`` keys such as:

.. code-block:: python

   from OpenPinch.lib.schemas.io import TargetInput

   payload = TargetInput.model_validate(
       {
           "streams": [...],
           "utilities": [...],
           "options": {
               "HENS_APPROACH_TEMPERATURES": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
               "HENS_DERIVATIVE_THRESHOLDS": [
                   0.5,
                   0.9,
                   1.3,
                   1.7,
                   2.1,
                   2.4,
                   2.8,
                   3.2,
                   3.6,
                   4.0,
               ],
               "HENS_METHOD_SEQUENCE": [
                   "pinch_decomposition",
                   "topology_design",
                   "energy_stage_refinement",
               ],
               "HENS_PDM_SOLVER": "couenne",
               "HENS_TDM_SOLVER": "couenne",
               "HENS_ESM_SOLVER": "ipopt-pyomo",
               "HENS_SOLVE_TOLERANCE": 1e-3,
               "HENS_MAX_PARALLEL": 10,
               "HENS_OUTPUT_FORMATS": ["json", "csv"],
               "HENS_RUN_ID": "example-run",
           },
       }
   )

   problem = PinchProblem(source=payload, project_name="Four-stream example")
   design = problem.design.heat_exchanger_network_synthesis()

Do not pass heat exchanger network design-space or solver controls as a separate object to the
design call. The call may receive non-design runtime state options, but
persistent heat exchanger network controls belong in the loaded problem payload.

Workspace Workflow
------------------

Use :class:`~OpenPinch.PinchWorkspace` when a named study needs variants,
comparisons, or bundle persistence. The workspace dispatches heat exchanger network synthesis to
the active variant's live ``PinchProblem.design`` path.

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace(
       source="Four-stream-Yee-and-Grossmann-1990-1.json",
       project_name="Four-stream converted OpenHENS example",
   )

   view = workspace.solve_variant(
       "baseline",
       workflow="heat_exchanger_network_synthesis",
   )

   problem = workspace.case("baseline")
   design = problem.results.design

``PinchWorkspace`` is the multi-case owner. Heat exchanger network synthesis
does not add a public case, study, scenario, or heat exchanger network-specific
workspace root.

Results and Optional Exports
----------------------------

The canonical in-memory result is ``problem.results``. Its
``TargetOutput.design`` field contains a
:class:`~OpenPinch.lib.schemas.synthesis.HeatExchangerNetworkSynthesisResult`
with objective values, task metadata, an optional manifest, and a
:class:`~OpenPinch.classes.heat_exchanger_network.HeatExchangerNetwork`.

The network exposes source/sink stream links through
:class:`~OpenPinch.classes.heat_exchanger.HeatExchanger` records. Recovery
links are hot process stream to cold process stream, hot utility links are hot
utility to cold process stream, and cold utility links are hot process stream
to cold utility.

OpenHENS-style network grid diagrams are available directly from the design
result. The rank is 1-based and follows the same accepted-method ordering used
for saved best-network candidates:

.. code-block:: python

   diagram = problem.results.design.grid_diagram(solution_rank=1)
   diagram.show()
   diagram.save("network.png")

The returned object exposes the Plotly ``fig`` object, a lightweight drawing
adapter on ``ax`` for tests and introspection, the selected ``network``, and the
normalized ``grid_model`` used to draw the process-stream topology. Saving to
``.png`` uses Plotly static image export; saving to ``.html`` writes an
interactive Plotly document.

The default layout follows the OpenHENS stage-based grid. Two optional keyword
arguments tune the diagram without changing the selected solution:

.. code-block:: python

   design.grid_diagram(stream_line_width=5.0)
   design.grid_diagram(temperature_scaled=True)

``stream_line_width`` controls the stream strokes and is also used to auto-size
the figure height for larger networks. ``temperature_scaled=True`` positions
stream and match x-coordinates on a high-to-low temperature axis while keeping
the same hot and cold stream direction conventions.

JSON and CSV files are optional export views generated from
``problem.results`` after synthesis:

.. code-block:: python

   from OpenPinch.services.heat_exchanger_network_synthesis.exports import (
       export_heat_exchanger_network_synthesis_results,
   )

   manifest = export_heat_exchanger_network_synthesis_results(
       problem,
       "results/Four-stream-Yee-and-Grossmann-1990-1",
       workspace_variant="baseline",
   )

The export helper writes views such as ``manifest.json``,
``results/<task_id>.json``, ``metrics/solution_metrics.csv``, and
``metrics/run_summary.csv``. Those files are exports from the result cache, not
the terminal output of the core workflow.

OpenHENS Name Mapping
---------------------

This mapping is documentation-only. OpenPinch does not provide runtime import
aliases, OpenHENS field aliases, command parity, or an ``OpenHENS`` facade.

.. list-table::
   :header-rows: 1

   * - OpenHENS concept
     - OpenPinch-native replacement
   * - ``CaseStudy.from_csv(...)``
     - Convert the source CSV example once into OpenPinch JSON or native
       ``TargetInput``; load it through ``PinchProblem``.
   * - ``SynthesisStudy``
     - ``PinchProblem`` for one prepared problem, or ``PinchWorkspace`` for
       named variants.
   * - ``DesignSpace``
     - ``HENS_APPROACH_TEMPERATURES`` and
       ``HENS_DERIVATIVE_THRESHOLDS`` in ``TargetInput.options``.
   * - ``MethodSequence.standard_pdm_tdm_esm()``
     - ``HENS_METHOD_SEQUENCE`` in ``TargetInput.options``.
   * - ``SolveSetup.local(...)``
     - ``HENS_PDM_SOLVER``, ``HENS_TDM_SOLVER``, ``HENS_ESM_SOLVER``,
       ``HENS_SOLVE_TOLERANCE``, and ``HENS_MAX_PARALLEL`` in
       ``TargetInput.options``.
   * - ``StudyOutputs``
     - ``HENS_OUTPUT_FORMATS``, ``HENS_OUTPUT_FOLDER``, and optional exports
       generated from ``problem.results``.
   * - ``OpenHENS(study).solve()``
     - ``problem.design.heat_exchanger_network_synthesis(...)`` or
       ``workspace.solve_variant(..., workflow="heat_exchanger_network_synthesis")``.
   * - ``NetworkSolution``
     - ``TargetOutput.design.network`` as ``HeatExchangerNetwork``.
   * - Study artifact directory
     - Optional JSON/CSV export views generated from ``problem.results`` and
       identified by OpenPinch problem or workspace variant identity.

Dependencies and Solver Tests
-----------------------------

Core ``import OpenPinch`` remains lightweight. Install heat exchanger network synthesis runtime
dependencies explicitly when live solver-backed synthesis is needed:

.. code-block:: bash

   python -m pip install "openpinch[synthesis]"

Repository development uses the same optional extra in an editable checkout:

.. code-block:: bash

   python -m pip install -e ".[synthesis]"

The ``synthesis`` extra installs Python packages such as Pyomo, GEKKO, IDAES,
plotting/export libraries, and wake-management tooling. External solver
binaries such as Couenne and IPOPT are installed separately. OpenPinch first
checks ``PATH`` and then the IDAES extension directory reported by
``idaes.bin_directory``; when a solver is found there, OpenPinch prepends that
directory to the process ``PATH`` so downstream Pyomo solver factories can
resolve the executable. Missing optional Python packages raise
``MissingSynthesisDependencyError`` with the ``openpinch[synthesis]`` install
path. Missing external executables raise ``MissingSynthesisSolverError`` with
the missing binary name and solver-test guidance.

Use the documented test tiers:

.. code-block:: bash

   pytest -m "not synthesis and not solver"
   pytest -m synthesis
   pytest -m solver

The dependency policy for maintainers is recorded in
:doc:`../developer/synthesis-dependency-policy`.
