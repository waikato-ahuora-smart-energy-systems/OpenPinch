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

.. automodule:: OpenPinch.utils.wkbook_to_json
   :members:

.. automodule:: OpenPinch.utils.csv_to_json
   :members:

.. automodule:: OpenPinch.utils.input_validation
   :members:

Export and Reporting
--------------------

Use the export helpers when you want the solved
:class:`~OpenPinch.lib.schema.TargetOutput` and
:class:`~OpenPinch.classes.zone.Zone` hierarchy written back to an Excel
workbook for review or archiving.

.. automodule:: OpenPinch.utils.export
   :members:

Math, Optimisation, and Targeting Helpers
-----------------------------------------

These modules provide reusable numerical support for targeting and post-
processing tasks.

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

Plotting and Instrumentation
----------------------------

These helpers cover quick standalone plotting, execution-time measurement, and
the richer Streamlit dashboard components used by ``streamlit_app.py``.

.. automodule:: OpenPinch.utils.plots
   :members:

.. automodule:: OpenPinch.utils.decorators
   :members:

.. automodule:: OpenPinch.streamlit_webviewer.web_graphing
   :members:
