Workflow Map
============

OpenPinch supports several valid entrypoints. The right one depends on whether
your problem is exploratory, repeatable, or embedded inside another
application.

Recommended Entry Points
------------------------

CLI workflow
   Use this when you want a quick solve, validation pass, graph export, or
   packaged sample/notebook copy operation without writing Python.

``PinchProblem`` workflow
   Use this for notebooks and scripts when you want one object that owns the
   problem definition, solved result, graph accessors, summaries, and exports.

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

- you have a file-backed case already
- you want a quick answer or export artifact
- you are validating sample workflows

Use `PinchProblem` when:

- you are working in notebooks or scripts
- you want summaries, graphs, exports, and scenario helpers on one object
- you want access to `problem.plot.*` and `problem.target.*`

Use `pinch_analysis_service` when:

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

   CLI / notebooks / scripts / external app
                    |
                    v
        PinchProblem or pinch_analysis_service
                    |
                    v
            data_preprocessing_service
                    |
                    v
                 Zone tree
                    |
                    v
        direct / indirect / HPR / cogeneration services
                    |
                    v
         TargetOutput + summaries + graphs + exports

Typical User Paths
------------------

First-time user
   `sample case -> validate -> run -> summary -> graph`

Notebook user
   `copy sample case -> PinchProblem -> run -> summary_frame -> plot`

Advanced HPR study
   `base case -> problem.target.direct_heat_pump(...) -> compare targets and GCC`

Programmatic integration
   `TargetInput schema -> pinch_analysis_service(...) -> TargetOutput`

Where To Go Next
----------------

- For runnable first workflows, go to :doc:`../guides/index`.
- For the thermodynamic model behind these workflows, go to
  :doc:`../fundamentals/index`.
- For the exact callable surfaces, go to :doc:`../api/index`.
