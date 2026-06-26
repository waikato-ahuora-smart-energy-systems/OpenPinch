What Is OpenPinch?
==================

OpenPinch is a process-integration toolkit for thermal targeting studies. It
combines classical pinch analysis workflows with Total Site utility
integration, graph generation, and optional advanced workflows such as heat
pump and refrigeration screening, exergy post-processing, plus turbine
cogeneration analysis.

The package is built around one numerical engine but exposed through several
distinct user surfaces:

- :class:`OpenPinch.PinchProblem` for one case at a time
- :class:`OpenPinch.PinchWorkspace` for named case studies and bundle
  persistence
- :func:`OpenPinch.main.pinch_analysis_service` for typed request/response
  integration
- :mod:`OpenPinch.resources` plus ``openpinch notebook`` for packaged learning
  assets

Technical Scope
---------------

OpenPinch answers questions such as:

- What are the minimum hot and cold utility targets for this process?
- How much heat can be recovered internally before utilities are needed?
- How does a direct process-level answer differ from an indirect site-level
  answer?
- Which graph best explains the remaining utility load?
- Does a candidate Heat Pump integration scenario improve the plant utility
  picture?
- What does the same solved target look like in exergy terms?
- How much above Pinch or below Pinch turbine work is theoretically available?

Primary Product Shape
---------------------

At a high level, the codebase turns validated inputs into a solved target set:

.. code-block:: text

   TargetInput / JSON / Excel / CSV
               |
               v
     validation + normalization
               |
               v
       prepared Zone hierarchy
               |
               +--> direct heat integration
               +--> indirect / Total Site targeting
               +--> HPR targeting
               +--> exergy post-processing
               +--> cogeneration post-processing
               |
               v
   TargetOutput + summaries + graphs + export data

The same underlying analysis engine can be reached through:

- :class:`OpenPinch.PinchProblem`
- :class:`OpenPinch.PinchWorkspace`
- :func:`OpenPinch.main.pinch_analysis_service`
- lower-level service helpers under :mod:`OpenPinch.services`
- packaged sample cases and notebooks

Who This Documentation Serves
-----------------------------

Thermal systems users
   Engineers and researchers who need the thermodynamic basis, workflow
   meaning, and output interpretation.

Python users
   Notebook and script users who need clear public entrypoints and example
   workflows.

Integrators and contributors
   Users embedding OpenPinch into larger software or extending the package
   internals.

What The Codebase Treats As Public
----------------------------------

For most users, the supported public surfaces are:

- :class:`OpenPinch.PinchProblem`
- :class:`OpenPinch.PinchWorkspace`
- :func:`OpenPinch.main.pinch_analysis_service`
- :mod:`OpenPinch.resources`
- the ``openpinch notebook`` CLI command

The CLI is intentionally small. It copies notebooks only. The actual solve,
graph export, Excel export, validation, and advanced workflow selection happen
through Python.

Advanced users can drop into schemas, service entrypoints, prepared `Zone`
trees, or lower-level analysis helpers when they need more control, but those
surfaces should be read with the support levels explained in
:doc:`support-and-stability`.

What OpenPinch Does Not Assume
------------------------------

OpenPinch does not assume that every study wants the same depth of control.
You can stay at the wrapper-object level, move down to a typed service
boundary, or inspect and mutate the in-memory zone tree directly.

Next Steps
----------

- Use :doc:`capability-matrix` for the package feature map.
- Use :doc:`workflow-map` to choose the right entrypoint.
- Use :doc:`../fundamentals/index` for the technical grounding.
- Use :doc:`../guides/index` for runnable workflows.
