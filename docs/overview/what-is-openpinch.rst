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

External Contract and Internal Tools
------------------------------------

``OpenPinch.main.pinch_analysis_service``
   The sole compatibility-protected Python contract. It accepts the supported
   request mapping and returns the structured targeting result.

``PinchProblem`` and ``PinchWorkspace``
   Unsupported internal application coordinators used by advanced notebooks,
   repository applications, and contributor workflows. Their paths and
   methods may change without a compatibility layer.

``OpenPinch.resources`` and ``openpinch notebook``
   Repository learning and asset-copy tooling. These are maintained, but are
   separate from the protected Python contract.

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

What Is Stable
--------------

For application integration, import
:func:`OpenPinch.main.pinch_analysis_service`, pass a request mapping, and
consume its structured return value. That is the only Python path covered by
the current compatibility policy.

Advanced workflows remain documented for development and research. They use
concrete owner modules deliberately and should be adopted only when internal
API churn is acceptable.

What OpenPinch Is Not
---------------------

OpenPinch is not a CLI-first solver. The CLI copies notebooks only.

OpenPinch is not a detailed exchanger mechanical design tool. HEN synthesis,
MVR, HPR, cogeneration, and exergy surfaces are targeting and screening
workflows that should be interpreted with the thermodynamic assumptions in the
case input and configuration.

Next Steps
----------

- :doc:`workflow-map` to distinguish the supported contract from internal owners.
- :doc:`capability-matrix` to see feature status.
- :doc:`../getting-started` for the shortest supported solve.
