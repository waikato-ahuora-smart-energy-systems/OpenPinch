Utilities and I/O
=================

These adapter, numerical, optimisation, and presentation modules are
unsupported contributor APIs. They are documented to aid maintenance, not as
compatibility promises.

The utilities layer supports the main targeting workflow with import/export
helpers, physical-property utilities, optimisation routines, and plotting
support. Several of these modules are also useful independently of the full
OpenPinch workflow.

Data Conversion and Validation
------------------------------

These modules translate spreadsheet- or table-oriented data sources into the
schema-compatible structures consumed by the service layer.

.. automodule:: OpenPinch.adapters
   :no-members:

.. automodule:: OpenPinch.adapters.io.workbook
   :members:

.. automodule:: OpenPinch.adapters.io.csv
   :members:

.. automodule:: OpenPinch.adapters.io.records
   :members:

Export and Reporting
--------------------

Use the export helpers when you want the solved
:class:`~OpenPinch.contracts.output.TargetOutput` and
:class:`~OpenPinch.domain.zone.Zone` hierarchy written back to an Excel
workbook for review or archiving.

.. automodule:: OpenPinch.presentation.reporting.workbook
   :members:

Math, Optimisation, and Utility Helpers
---------------------------------------

These modules provide reusable numerical support for targeting and post-
processing tasks.

.. automodule:: OpenPinch.analysis.numerics
   :members:

.. automodule:: OpenPinch.optimisation.service
   :members:

.. automodule:: OpenPinch.optimisation.models
   :members:

.. automodule:: OpenPinch.domain._stream.linearisation
   :members:

.. automodule:: OpenPinch.analysis.heat_transfer
   :members:

.. automodule:: OpenPinch.analysis.economics
   :members:

.. automodule:: OpenPinch.analysis.thermodynamics.water
   :members:

Internal Optimiser Backends
---------------------------

The modules below back
:func:`OpenPinch.optimisation.service.run_multistart_minimisation`. They are
primarily useful when inspecting or extending optimiser implementations.

.. automodule:: OpenPinch.optimisation.backends
   :no-members:

.. automodule:: OpenPinch.optimisation.candidates

.. automodule:: OpenPinch.optimisation.backends.dual_annealing

.. automodule:: OpenPinch.optimisation.backends.cma_es

.. automodule:: OpenPinch.optimisation.backends.bayesian

.. automodule:: OpenPinch.optimisation.backends.rbf

Plotting and Instrumentation
----------------------------

These helpers cover quick standalone plotting, execution-time measurement, and
the richer Streamlit dashboard components used by ``streamlit_app.py``.

Install ``openpinch[notebook]`` for the standalone Plotly plotting helpers and
Excel-oriented utility modules. Install ``openpinch[dashboard]`` for the
Streamlit dashboard path.

.. automodule:: OpenPinch.presentation.graphs.simple
   :members:

.. automodule:: OpenPinch.application
   :members:

.. automodule:: OpenPinch.presentation.dashboard.rendering
   :members:
