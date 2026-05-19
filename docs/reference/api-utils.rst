Utilities and I/O
=================

The utilities layer supports the main targeting workflow with import/export
helpers, physical-property utilities, optimisation routines, and plotting
support. Several of these modules are also useful independently of the full
OpenPinch workflow.

Data Conversion and Validation
------------------------------

These modules translate spreadsheet- or table-oriented data sources into the
schema-compatible structures consumed by the service layer.

.. automodule:: OpenPinch.utils
   :no-members:

.. automodule:: OpenPinch.utils.wkbook_to_json
   :members:

.. automodule:: OpenPinch.utils.csv_to_json
   :members:

.. automodule:: OpenPinch.utils.input_validation
   :members:

Export and Reporting
--------------------

Use the export helpers when you want the solved
:class:`~OpenPinch.lib.schemas.io.TargetOutput` and
:class:`~OpenPinch.classes.zone.Zone` hierarchy written back to an Excel
workbook for review or archiving.

.. automodule:: OpenPinch.utils.export
   :members:

Math, Optimisation, and Targeting Helpers
-----------------------------------------

These modules provide reusable numerical support for targeting and post-
processing tasks.

.. automodule:: OpenPinch.utils.multiscale_targeting
   :members:

.. automodule:: OpenPinch.utils.miscellaneous
   :members:

.. automodule:: OpenPinch.utils.blackbox_minimisers
   :members:

.. automodule:: OpenPinch.utils.stream_linearisation
   :members:

.. automodule:: OpenPinch.utils.heat_exchanger
   :members:

.. automodule:: OpenPinch.utils.costing
   :members:

.. automodule:: OpenPinch.utils.water_properties
   :members:

Internal Optimiser Backends
---------------------------

The modules below back :func:`OpenPinch.utils.blackbox_minimisers.multiminima`.
They are primarily useful when you need to inspect or extend the optimiser
implementations themselves.

.. automodule:: OpenPinch.utils.bb_optimisers
   :no-members:

.. automodule:: OpenPinch.utils.bb_optimisers.common

.. automodule:: OpenPinch.utils.bb_optimisers.dual_annealing

.. automodule:: OpenPinch.utils.bb_optimisers.cmaes

.. automodule:: OpenPinch.utils.bb_optimisers.bayesian_optimisation

.. automodule:: OpenPinch.utils.bb_optimisers.rbf_surrogate

Plotting and Instrumentation
----------------------------

These helpers cover quick standalone plotting, execution-time measurement, and
the richer Streamlit dashboard components used by ``streamlit_app.py``.

Install ``openpinch[notebook]`` for the standalone Plotly plotting helpers and
Excel-oriented utility modules. Install ``openpinch[dashboard]`` for the
Streamlit dashboard path.

.. automodule:: OpenPinch.utils.plots
   :members:

.. automodule:: OpenPinch.utils.decorators
   :members:

.. automodule:: OpenPinch.streamlit_webviewer.web_graphing
   :members:
