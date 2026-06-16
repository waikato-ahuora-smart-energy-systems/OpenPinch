Heat Exchanger Network Synthesis
================================

Heat exchanger network synthesis is exposed through the same problem and
workspace roots as the rest of OpenPinch. A HEN run starts from an
OpenPinch-compatible JSON payload, a native
:class:`~OpenPinch.lib.schemas.io.TargetInput`, or an already loaded
:class:`~OpenPinch.PinchProblem`. Source OpenHENS CSV files are migration
source material only; convert them once into OpenPinch JSON or native
``TargetInput`` payloads before running synthesis.

The implementation boundary is
``heat_exchanger_network_synthesis_service(problem)``. It is internal and
problem-rooted: it requires a live ``PinchProblem`` and reads persistent HEN
configuration from ``TargetInput.options`` through the prepared
``Configuration``. User code should call the problem design accessor or the
workspace workflow dispatch shown below, not the internal service directly.

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

Do not pass HEN design-space or solver controls as a separate object to the
design call. The call may receive non-design runtime state options, but
persistent HEN controls belong in the loaded problem payload.

Workspace Workflow
------------------

Use :class:`~OpenPinch.PinchWorkspace` when a named study needs variants,
comparisons, or bundle persistence. The workspace dispatches HEN synthesis to
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

``PinchWorkspace`` is the multi-case owner. HEN synthesis does not add a public
case, study, scenario, or HEN-specific workspace root.

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

Core ``import OpenPinch`` remains lightweight. Install HEN synthesis runtime
dependencies explicitly when live solver-backed synthesis is needed:

.. code-block:: bash

   python -m pip install "openpinch[synthesis]"

Repository development uses the same optional extra through uv:

.. code-block:: bash

   rtk uv sync --extra synthesis

The ``synthesis`` extra installs Python packages such as Pyomo, GEKKO,
plotting/export libraries, and wake-management tooling. External solver
binaries such as Couenne and IPOPT are installed separately and must be
available on ``PATH`` for marked solver tests. Missing optional Python packages
raise ``MissingSynthesisDependencyError`` with the ``openpinch[synthesis]``
install path. Missing external executables raise ``MissingSynthesisSolverError``
with the missing binary name and solver-test guidance.

Use the documented test tiers:

.. code-block:: bash

   rtk uv run pytest -m "not synthesis and not solver"
   rtk uv run pytest -m synthesis
   rtk uv run pytest -m solver

The dependency policy for maintainers is recorded in
:doc:`../developer/synthesis-dependency-policy`.
