:orphan:

Quickstart Workflow
===================

This page has been replaced by the new guide set, but the current quickstart
surface is still:

- install with ``python -m pip install openpinch``
- add notebook support with ``python -m pip install "openpinch[notebook]"``
- add dashboard support with ``python -m pip install "openpinch[dashboard]"``
- solve with ``PinchProblem.target()``
- inspect ``summary_frame()``
- graph with ``plot.grand_composite_curve()``
- export with ``export_excel()`` and ``problem.plot.export(...)``
- review advanced heat pump and refrigeration routes with
  ``problem.target.direct_heat_pump(...)``
- use ``problem.target.direct_heat_integration(period_id="peak")`` or
  ``problem.target.indirect_heat_integration(period_id="winter")`` when the
  payload is period-valued
- use ``show_dashboard()`` for the Streamlit dashboard
- use ``PinchWorkspace`` for named baseline-versus-variant studies

Packaged notebooks:

- ``01_basic_pinch_and_dtcont_sensitivity.ipynb``
- ``02_total_site_targets_and_sugcc.ipynb``
- ``03_carnot_hpr_comparison.ipynb``
- ``04_multiperiod_targeting_and_period_comparison.ipynb``
- ``05_schema_service_and_output_workflows.ipynb``
- ``06_energy_transfer_analysis.ipynb``
- ``07_vapour_compression_mvr_cascade_hpr.ipynb``
- ``08_direct_gas_stream_mvr.ipynb``
- ``09_hen_design_service_four_stream.ipynb``

Use these pages instead:

- :doc:`../guides/first-solve-python`
- :doc:`../guides/input-formats-and-validation`
- :doc:`../guides/notebooks-and-sample-cases`
- :doc:`../api/pinchproblem`
