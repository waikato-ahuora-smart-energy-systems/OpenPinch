Heat Exchanger Network Synthesis
================================

Heat exchanger network synthesis is exposed through the same problem and
workspace roots as the rest of OpenPinch. A heat exchanger network run starts from an
OpenPinch-compatible JSON case input, a native
:class:`~OpenPinch.lib.schemas.io.TargetInput`, or an already loaded
:class:`~OpenPinch.PinchProblem`. Source OpenHENS CSV files are migration
source material only; convert them once into OpenPinch JSON or native
``TargetInput`` models before running synthesis.

The service ingress is
``heat_exchanger_network_synthesis_entry.py``. It is internal and
problem-rooted: it requires a live ``PinchProblem`` and reads persistent heat
exchanger network configuration from ``TargetInput.options`` through the
prepared ``Configuration``. User code should call
``problem.design.enhanced_synthesis_method(quality_tier=...)``,
``problem.design.open_hens_method(...)``, or
``problem.design.heat_exchanger_network_synthesis(method=...)`` instead of the
internal service directly. Those public calls invoke the local solver-backed
synthesis executor by default.

When Couenne is unavailable for the Couenne-backed derivative/topology stages,
OpenPinch emits a warning and attempts ``network_evolution_method`` directly
with the configured EVM solver and stage selection.

Design Method Accessors
-----------------------

Heat exchanger network synthesis has one umbrella accessor, one enhanced
quality-tier accessor, and four explicit design-method accessors:

.. code-block:: python

   from OpenPinch.lib import HENDesignMethod

   # Fast generic default: tier 0 compact PDM -> EVM.
   problem.design.heat_exchanger_network_synthesis()

   # Primary quality-tier entrypoint. Tier 2 is the enhanced baseline.
   problem.design.enhanced_synthesis_method(quality_tier=2)

   # Original OpenHENS sequence: tier 1 PDM -> TDM -> EVM.
   problem.design.open_hens_method()
   problem.design.heat_exchanger_network_synthesis(
       method=HENDesignMethod.OpenHENS,
   )

   # Individual method calls.
   problem.design.pinch_design_method()
   problem.design.thermal_derivative_method(initial_networks=(seed_network,))
   problem.design.network_evolution_method(initial_networks=(existing_network,))

``HENDesignMethod`` is an alias for
:class:`~OpenPinch.lib.enums.HeatExchangerNetworkDesignMethod`. The enum values
are the canonical method identifiers stored in tasks, manifests, and results:

.. list-table::
   :header-rows: 1

   * - Enum member
     - Stored identifier
     - Meaning
   * - ``HENDesignMethod.OpenHENS``
     - ``"open_hens_method"``
     - original tier-1 OpenHENS sequence, ``pinch_design_method -> thermal_derivative_method -> network_evolution_method``
   * - ``HENDesignMethod.PinchDesign``
     - ``"pinch_design_method"``
     - pinch design method only
   * - ``HENDesignMethod.ThermalDerivative``
     - ``"thermal_derivative_method"``
     - thermal derivative method only
   * - ``HENDesignMethod.NetworkEvolution``
     - ``"network_evolution_method"``
     - network evolution method only

The umbrella accessor dispatches to the same services as the direct accessors
when a method is supplied. With no method it uses the fast tier 0 OpenHENS
route:

.. code-block:: python

   problem.design.heat_exchanger_network_synthesis(
       method=HENDesignMethod.PinchDesign,
   )
   problem.design.heat_exchanger_network_synthesis(
       method=HENDesignMethod.ThermalDerivative,
       initial_networks=(seed_network,),
   )
   problem.design.heat_exchanger_network_synthesis(
       method=HENDesignMethod.NetworkEvolution,
       initial_networks=(existing_network,),
   )

Seed behavior is explicit. ``pinch_design_method`` does not accept an initial
network. ``thermal_derivative_method`` requires either ``initial_networks`` or a
cached accepted ``pinch_design_method`` result on the problem. Likewise,
``network_evolution_method`` requires either ``initial_networks`` or a cached
accepted ``thermal_derivative_method`` result. Passing an already existing
:class:`~OpenPinch.classes.heat_exchanger_network.HeatExchangerNetwork` to
``network_evolution_method(initial_networks=...)`` is the retrofit path: the
method evolves that seed network into ranked candidate networks without first
running pinch design or thermal derivative tasks.

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

   design = problem.design.open_hens_method()

   result = problem.results
   network = result.design.network

   for exchanger in network.exchangers:
       print(exchanger.source_stream, "->", exchanger.sink_stream, exchanger.duty)

The ``source`` file above is an OpenPinch-compatible JSON case input. The original
OpenHENS CSV example is not loaded at runtime by synthesis. Its process streams
map to ``StreamSchema`` records, utilities and utility prices map to
``UtilitySchema`` records, and the synthesis controls map to ``TargetInput``
``options`` keys such as:

.. code-block:: python

   from OpenPinch.lib.schemas.io import TargetInput

   target_input = TargetInput.model_validate(
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
                   "pinch_design_method",
                   "thermal_derivative_method",
                   "network_evolution_method",
               ],
               "HENS_SYNTHESIS_QUALITY_TIER": 1,
               "HENS_SOLVER_PDM": "couenne",
               "HENS_SOLVER_TDM": "couenne",
               "HENS_SOLVER_EVM": "ipopt-pyomo",
               "HENS_SOLVE_TOLERANCE": 1e-3,
               "HENS_MAX_PARALLEL": 10,
               "HENS_OUTPUT_FORMATS": ["json", "csv"],
               "HENS_RUN_ID": "example-run",
           },
       }
   )

   problem = PinchProblem(source=target_input, project_name="Four-stream example")
   design = problem.design.open_hens_method()

Do not pass heat exchanger network design-space or solver controls as a separate object to the
design call. The call may receive non-design runtime context options, but
persistent heat exchanger network controls belong in the loaded problem input.

Quality Tiers
-------------

``problem.design.enhanced_synthesis_method(quality_tier=...)`` is the primary
public way to run OpenHENS quality tiers. ``quality_tier`` accepts values
``0`` through ``5`` and defaults to tier 2. ``open_hens_method()`` always runs
the original tier 1 sequence. ``heat_exchanger_network_synthesis()`` with no
method runs the fast tier 0 route.

``HENS_SYNTHESIS_QUALITY_TIER`` remains a persistent configuration field for
advanced prepared-problem workflows. Its configuration default is tier 1, but
the public method-level entrypoints above supply call-local tier choices.
Higher tiers add protected fallback routes and quality-search routes; they do
not replace tier 1 unless a candidate gives a better valid network.

.. list-table::
   :header-rows: 1

   * - Tier
     - Meaning
     - Default generated routes
   * - ``0``
     - Fast compact screening
     - compact PDM -> EVM at multiplier ``1.0``
   * - ``1``
     - Original OpenHENS
     - standard PDM -> TDM -> EVM
   * - ``2``
     - Base quality search
     - protected tiers 0 and 1 plus compact/direct and raw/TDM routes at
       multiplier ``1.0``
   * - ``3``
     - dTmin sweep
     - tier 2 plus compact/direct and raw/TDM routes at multipliers
       ``1.0`` and ``2.0``
   * - ``4``
     - Balanced quality search
     - tier 3 routes with EVM add/remove branch breadth ``2/2``
   * - ``5``
     - Experimental broad dTmin search
     - tier 4 breadth with multipliers ``0.5``, ``1.0``, and ``2.0``

The generated multiplier is applied to the first configured
``HENS_APPROACH_TEMPERATURES`` value. ``HENS_DT_CONT_MULTIPLIERS`` is an expert
override for non-standard tier routes and is always interpreted as a
multiplier, not an absolute approach temperature. ``HENS_EVM_N_AD_BRANCHES``
and ``HENS_EVM_N_RM_BRANCHES`` override tier-derived EVM branch breadth when
supplied.

Compact PDM routes go directly to network evolution. Raw and standard routes
are the routes that use TDM. Tiers 2 and above preserve protected tier 0 and
tier 1 candidates so a broader search can fall back to the fast or original
OpenHENS result when the additional routes do not improve the objective.

Robustness and Topology Normalization
-------------------------------------

The enhanced tiers keep the original OpenHENS route available while adding
routes that are deliberately smaller or more redundant:

- Compact PDM routes build a reduced task surface and pass successful
  topologies directly to EVM. They avoid TDM when the derivative stage is not
  needed for that pathway.
- Raw PDM routes and the original standard route are the only generated routes
  that use TDM.
- Tier 4 and tier 5 increase EVM add/remove branch breadth to ``2/2`` and use
  no-improvement pruning so branch search can recover from one poor topology
  edit without continuing indefinitely.

Topology restrictions are canonicalized before downstream quality-route tasks
are built. Empty recovery-stage gaps are removed, independent matches that were
packed into one original stage are split into consecutive canonical stages, and
matches that share a hot or cold process stream remain in the same stage. This
makes equivalent grid layouts compare as the same topology and reduces
duplicate seeded EVM tasks.

Compact PDM routes use recovery-stage packing, and advanced workflows can
enable the same packing for standard PDM and/or TDM with ``HENS_STAGE_PACKING``.
Those constraints force active recovery stages to be contiguous and bind active
matches to a small positive duty threshold. The practical goal is to reduce
stage-index symmetry in the stage-wise superstructure, which makes equivalent
solutions less likely to appear as different solver choices and gives
downstream topology evolution a cleaner starting point.

Benchmarking Quality Tiers
--------------------------

``scripts/benchmark_performance.py`` is the non-gating benchmark harness used
to compare small OpenHENS fixture cases. Its default case set covers unique
fixtures up to nine process streams, excludes reordered duplicates, and runs
production-supported tiers ``0`` through ``4``. Tier ``5`` is intentionally
documented as experimental because its broader dTmin sweep adds runtime cost
and is better suited to research or benchmarking runs.

The benchmark output is incremental JSON. It records total runtime, task and
stage counts, selected pathway metadata, failure categories, and partial
diagnostics for timed-out or interrupted runs. Use it for local performance
comparison rather than as a replacement for the unit and contract tests.

Result Metadata and Contracts
-----------------------------

Each public method validates its task input and task output through the shared
Pydantic models in :mod:`OpenPinch.lib.schemas.synthesis`. The shared method
input includes the run identity, selected method, problem/workspace metadata,
settings, optional seed network, optional seed-network index, and trace
metadata. The shared method output includes the method, status, accepted
networks, ranked networks, task manifest, diagnostics, and trace metadata.

``HeatExchangerNetworkSynthesisResult`` records both the high-level design
method and the selected task method. For a direct method call these usually
match. For OpenHENS, ``design.design_method`` is
``HENDesignMethod.OpenHENS``, while ``design.method`` records the method that
produced the selected accepted network, normally
``HENDesignMethod.NetworkEvolution``. The
``design.manifest.method_sequence`` field keeps the task-level method sequence
used to build the executed task graph. ``design.manifest.selected_pathway_id``
records which quality-tier pathway produced the accepted network, and
``selected_protected_pathway`` indicates whether that winner was one of the
protected fallback routes.

.. code-block:: python

   from OpenPinch.lib import HENDesignMethod

   design = problem.design.enhanced_synthesis_method(quality_tier=2)

   assert design.design_method == HENDesignMethod.OpenHENS
   assert design.manifest.design_method == HENDesignMethod.OpenHENS

   for outcome in design.ranked_networks:
       print(outcome.method, outcome.status, outcome.objective_value)

Method-Oriented Service Layout
------------------------------

The HEN synthesis implementation mirrors the public method boundary. The entry
module owns service dispatch, the ``targeting_services`` package owns
method-specific orchestration, ``common`` owns shared execution/results/solver
support, and ``unit_models`` owns the equation/unit model layer.

.. code-block:: text

   OpenPinch/services/heat_exchanger_network_synthesis/
     heat_exchanger_network_synthesis_entry.py
     targeting_services/
       open_hens_method.py
       pinch_design_method.py
       thermal_derivative_method.py
       network_evolution_method.py
       topology.py
     common/
       execution/
         pathways.py
       reporting/
       results/
       solver/
       service_context.py
       errors.py
     unit_models/
       base.py
       pinch_design.py
       stagewise.py
       packed_pinch_design.py
       packed_stagewise.py
       stage_packing.py
       problem.py

``open_hens_method.py`` is intentionally a composition layer for the OpenHENS
workflow. It calls the explicit method stages rather than
building tasks itself. The individual method files are where method-specific
task generation and stage execution live. Old import paths such as
``service.py``, ``methods.full_sequence``, ``solver.*``, ``equations.*``, and
``reporting.*`` are not compatibility shims; they have been removed.

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
with objective values, ranked network candidates, an optional manifest, and a
selected
:class:`~OpenPinch.classes.heat_exchanger_network.HeatExchangerNetwork`.

The selected network is available as ``design.network``. The public design
service ranks successful network candidates by objective value, removes
duplicate exchanger-connection structures, stores that unique list on
``design.ranked_networks``, and selects rank 1 by default:

.. code-block:: python

   design = problem.design.enhanced_synthesis_method(quality_tier=2)

   # Rank 1 is selected by default.
   selected = design.network
   ranked = design.ranked_networks

   assert ranked[0].network == selected

Use ``get_n_best_networks(...)`` when you only need the first few ranked
candidates:

.. code-block:: python

   top_three = design.get_n_best_networks(3)

   for rank, outcome in enumerate(top_three, start=1):
       print(rank, outcome.objective_value, outcome.task.task_id)

Use ``select_network(...)`` to make another ranked candidate the selected
network. The method mutates the design result, updates ``design.network`` and
the associated selected-task metadata, and returns the same design object:

.. code-block:: python

   design.select_network(solution_rank=2)

   print(design.task_id)
   print(design.objective_values)
   print(len(design.network.exchangers))

``solution_rank`` is 1-based. Requesting an unavailable rank raises
``IndexError`` with the number of available ranked networks.

The problem design accessor also exposes convenience totals for the currently
selected network through ``problem.design.network``:

.. code-block:: python

   problem.design.network.total_heat_recovery
   problem.design.network.total_hot_utility
   problem.design.network.total_cold_utility
   problem.design.network.utility("HU1")

These helpers read from ``problem.results.design.network``. Run a design method
such as ``problem.design.enhanced_synthesis_method(quality_tier=2)`` first so a
selected network is cached on the problem. ``utility(name)`` returns duty for
hot or cold utility exchangers whose utility stream identity matches ``name``.

The network exposes source/sink stream links through
:class:`~OpenPinch.classes.heat_exchanger.HeatExchanger` records. Recovery
links are hot process stream to cold process stream, hot utility links are hot
utility to cold process stream, and cold utility links are hot process stream
to cold utility.

OpenHENS-style network grid diagrams are available directly from the selected
:class:`~OpenPinch.classes.heat_exchanger_network.HeatExchangerNetwork`:

.. code-block:: python

   design = problem.results.design

   diagram = design.network.build_grid_diagram()
   diagram.show()
   diagram.save("network.png")

For synthesis results, ``grid_diagram(...)`` remains available as a ranked
convenience wrapper. The rank is 1-based and follows the same accepted-method
ordering used for ranked network candidates:

.. code-block:: python

   diagram = design.grid_diagram(solution_rank=1)

The returned object exposes the Plotly ``fig`` object, a lightweight drawing
adapter on ``ax`` for tests and introspection, the selected ``network``, and the
normalized ``grid_model`` used to draw the process-stream topology:

.. code-block:: python

   diagram.fig
   diagram.ax
   diagram.network
   diagram.grid_model

The Plotly figure includes hover information on exchanger markers, including
duty, area when available, and match description. Saving to ``.png`` uses Plotly
static image export; saving to ``.html`` writes an interactive Plotly document.

The default layout follows the OpenHENS stage-based grid. Two optional keyword
arguments tune the diagram without changing the selected network:

.. code-block:: python

   design.network.build_grid_diagram(stream_line_width=5.0)
   design.network.build_grid_diagram(temperature_scaled=True)

``stream_line_width`` controls the stream strokes and is also used to auto-size
the figure height for larger networks. The actual marker size and connector
line widths are derived from the available lane pitch so diagrams scale with
the number of process streams, utility matches, and stream split lanes.
``temperature_scaled=True`` positions stream and match x-coordinates on a
high-to-low temperature axis while keeping the same hot and cold stream
direction conventions.

The standalone service accepts multiple ranked network candidates directly. Use
the 0-based ``index`` argument to draw one specific network, or omit ``index``
to receive a tuple of diagrams:

.. code-block:: python

   from OpenPinch.services.network_grid_diagram import build_grid_diagram

   networks = tuple(
       outcome.network for outcome in design.get_n_best_networks()
   )

   build_grid_diagram(networks, index=0)
   build_grid_diagram(networks)

For the synthesis-result wrapper, ``solution_rank`` remains 1-based:

.. code-block:: python

   design.grid_diagram(solution_rank=1)
   design.grid_diagram(solution_rank=2)

Calling ``select_network(solution_rank=2)`` changes ``design.network`` for
subsequent result inspection, but it is not required before using the wrapper
with ``grid_diagram(solution_rank=2)``.

JSON and CSV files are optional export views generated from
``problem.results`` after synthesis:

.. code-block:: python

   from OpenPinch.services.heat_exchanger_network_synthesis.common.reporting.exports import (
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
     - ``HENS_METHOD_SEQUENCE`` in ``TargetInput.options`` and
       ``HENDesignMethod.OpenHENS`` for the public method dispatcher.
   * - Search quality/breadth controls
     - ``problem.design.enhanced_synthesis_method(quality_tier=...)`` for the
       public tier selector, with persistent ``HENS_SYNTHESIS_QUALITY_TIER``,
       optional ``HENS_DT_CONT_MULTIPLIERS``, optional ``HENS_STAGE_PACKING``,
       and optional EVM branch overrides in ``TargetInput.options`` for
       prepared-problem defaults.
   * - ``SolveSetup.local(...)``
     - ``HENS_SOLVER_PDM``, ``HENS_SOLVER_TDM``, ``HENS_SOLVER_EVM``,
       ``HENS_SOLVE_TOLERANCE``, and ``HENS_MAX_PARALLEL`` in
       ``TargetInput.options``.
   * - ``StudyOutputs``
     - ``HENS_OUTPUT_FORMATS``, ``HENS_OUTPUT_FOLDER``, and optional exports
       generated from ``problem.results``.
   * - ``OpenHENS(study).solve()``
     - ``problem.design.open_hens_method()``,
       ``problem.design.enhanced_synthesis_method(quality_tier=...)``, or
       ``workspace.solve_variant(..., workflow="heat_exchanger_network_synthesis")``.
   * - ``NetworkSolution``
     - ``TargetOutput.design.network`` as ``HeatExchangerNetwork``.
   * - Retrofit/evolve an existing network
     - ``problem.design.network_evolution_method(initial_networks=(network,))``.
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
