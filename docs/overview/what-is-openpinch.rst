What Is OpenPinch?
==================

OpenPinch is a process-integration toolkit for thermal targeting studies. It
combines Pinch Analysis, Total Site utility integration, graph interpretation,
and advanced screening workflows behind a Python-first interface.

Use OpenPinch when you need to answer questions such as:

- What are the minimum hot and cold utility targets for this process?
- How much heat recovery is available under the selected temperature approach?
- How does a process-level answer differ from a site-level utility-system answer?
- Which graph explains the remaining heating or cooling demand?
- Does a Heat Pump, refrigeration, MVR, exergy, cogeneration, or HEN synthesis
  workflow improve the study decision?

Primary User Surfaces
---------------------

``PinchProblem``
   One live case with loading, validation, targeting, summaries, graphs,
   exports, period selection, process-component mutation, and design accessors.

``PinchWorkspace``
   Named baseline-versus-variant studies, bundle persistence, serializable
   variant views, and comparisons over real ``PinchProblem`` cases.

``pinch_analysis_service``
   Typed ``TargetInput`` to ``TargetOutput`` execution for applications that
   do not need a live wrapper object.

``OpenPinch.resources`` and ``openpinch notebook``
   Packaged sample cases and notebooks for learning, regression examples, and
   reproducible demonstrations.

Product Shape
-------------

.. code-block:: text

   TargetInput / JSON / Excel / CSV / sample case
               |
               v
      validation + input normalization
               |
               v
        prepared Zone hierarchy
               |
               +--> direct integration
               +--> indirect / Total Site targeting
               +--> Heat Pump / refrigeration targeting
               +--> exergy and cogeneration post-processing
               +--> heat exchanger network synthesis
               |
               v
      TargetOutput + summaries + graph data + exports

What Is Stable First
--------------------

The most supported user path is:

1. load a case with ``PinchProblem`` or ``PinchWorkspace``
2. validate and target it
3. read ``summary_frame()``
4. inspect graph families through ``problem.plot``
5. export Excel or HTML graph artifacts if needed

Advanced workflows are documented, but they assume the base thermal picture is
already understood.

What OpenPinch Is Not
---------------------

OpenPinch is not a CLI-first solver. The CLI copies notebooks only.

OpenPinch is not a detailed exchanger mechanical design tool. HEN synthesis,
MVR, HPR, cogeneration, and exergy surfaces are targeting and screening
workflows that should be interpreted with the thermodynamic assumptions in the
case input and configuration.

Next Steps
----------

- :doc:`workflow-map` to choose the right public entrypoint.
- :doc:`capability-matrix` to see feature status.
- :doc:`../getting-started` for the shortest supported solve.
