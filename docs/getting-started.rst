:orphan:

Getting Started
===============

This page has been replaced by the new task-driven guides and overview pages,
but it still keeps the core install and workflow references that older
bookmarks and tests expect.

Install the base package with ``python -m pip install openpinch``.

Install the notebook workflow with
``python -m pip install "openpinch[notebook]"``.

Install the dashboard workflow with
``python -m pip install "openpinch[dashboard]"``.

Install the TESPy-backed Brayton-cycle workflow with
``python -m pip install "openpinch[brayton_cycle]"``.

The main Python wrappers are ``PinchProblem`` for single cases and
``PinchWorkspace`` for named multi-case studies. The current workflow names
include ``validation_report()``, ``target()``, ``summary_frame()``, ``report()``,
``metrics()``, ``export_excel()``, ``plot.grand_composite_curve()``,
``plot.export_gallery(...)``, ``show_dashboard()``,
``problem.target.direct_heat_pump(...)``, ``workspace.scenario(...)``, and
``workspace.compare_cases(...)``. When a case definition carries multiperiod
values, named ``problem.target.*`` workflows also accept ``period_id=...``.

Use these pages instead:

- :doc:`guides/first-solve-python`
- :doc:`guides/notebooks-and-sample-cases`
- :doc:`api/cli-and-resources`
- :doc:`overview/workflow-map`
