Workflow Map
============

OpenPinch supports several valid entrypoints, but they are not interchangeable.
The right one depends on what owns the study state, where the inputs come from,
and whether you need a live notebook object, a typed service contract, or just
the packaged learning assets.

Recommended Entry Points
------------------------

Notebook-copy CLI
   Use this when you want the packaged notebook series copied into your working
   directory. The CLI does not solve cases, validate data, or export results.

Packaged resource workflow
   Use :mod:`OpenPinch.resources` when you want packaged sample cases or
   notebooks from Python without relying on filesystem-relative paths.

``PinchProblem`` workflow
   Use this for notebooks and scripts when you want one object that owns the
   problem definition, validation, prepared zone tree, solved result, graph
   accessors, summaries, state selection, and exports.

``PinchWorkspace`` workflow
   Use this for notebooks and scripts when you want named study cases,
   baseline-versus-variant comparison, serializable variant views, and
   bundle save/load on top of real ``PinchProblem`` cases.

Service-layer workflow
   Use :func:`OpenPinch.main.pinch_analysis_service` when you want a typed
   request/response boundary for another application or automation layer.

Prepared-zone workflow
   Use :func:`OpenPinch.services.input_data_processing.data_preparation.prepare_problem`
   and the lower-level service helpers only when you need to inspect or mutate
   the intermediate `Zone` hierarchy directly.

Decision Guide
--------------

Use the CLI when:

- you want ``openpinch notebook -o notebooks``
- you want a maintained onboarding asset copied locally
- you do not need a solve yet

Use :mod:`OpenPinch.resources` when:

- you want to list, read, or copy packaged sample cases from Python
- you want code to be explicit about where examples come from
- you want notebook or sample-case assets without hard-coding repo paths

Use ``PinchProblem`` when:

- you are working in notebooks or scripts
- you want to load one case from JSON, Excel, CSV, or a packaged sample name
- you want validation, summaries, graphs, exports, and advanced targeting on
  one object
- you want access to ``problem.plot.*``, ``problem.target.*``,
  ``problem.state_ids``, or ``problem.target_all_states()``

Use ``PinchWorkspace`` when:

- you need named baseline and variant cases in one session
- you want to compare cases without rebuilding case-input management helpers
- you want both live ``PinchProblem`` cases and serializable variant views in
  the same study
- you want to persist the full study as a bundle

Use ``pinch_analysis_service`` when:

- you are integrating OpenPinch into another codebase
- you want a typed `TargetInput -> TargetOutput` boundary
- you do not need the convenience wrapper state model

Use the lower-level service stack when:

- you need direct access to the prepared `Zone` tree
- you want to separate validation, preparation, targeting, and extraction
- you are doing advanced or research-oriented post-processing

Workflow Layering
-----------------

.. code-block:: text

   notebooks / scripts / external app
                 |
                 +--> openpinch notebook
                 +--> OpenPinch.resources
                 |
                 v
   PinchWorkspace / PinchProblem / pinch_analysis_service
                 |
                 v
         data_preprocessing_service
                 |
                 v
              Zone tree
                 |
                 v
   direct / indirect / HPR / exergy / cogeneration services
                 |
                 v
      TargetOutput + summaries + graphs + exports

Typical User Paths
------------------

First-time user
   ``openpinch notebook -o notebooks -> run notebook 01 -> inspect real
   PinchWorkspace and PinchProblem calls``

Notebook user
   ``PinchWorkspace(source="crude_preheat_train.json") -> copy_case ->
   set_dt_cont_multiplier -> compare_cases``

Single-case script
   ``PinchProblem("basic_pinch.json") -> validate -> target -> summary_frame ->
   plot.grand_composite_curve()``

Stateful study
   ``problem.state_ids -> problem.target.direct_heat_integration(state_id="peak")
   -> problem.target_all_states()``

Advanced HPR study
   ``base case -> problem.target.direct_heat_pump(...) -> compare summary rows
   and GCC / HPR graph surfaces``

Advanced exergy study
   ``base case -> problem.target.exergy(...) -> inspect enriched target row and
   exergetic GCC / NLP surfaces``

Programmatic integration
   ``TargetInput schema -> pinch_analysis_service(...) -> TargetOutput``

Where To Go Next
----------------

- For runnable first workflows, go to :doc:`../guides/index`.
- For the thermodynamic model behind these workflows, go to
  :doc:`../fundamentals/index`.
- For the exact callable surfaces, go to :doc:`../api/index`.
