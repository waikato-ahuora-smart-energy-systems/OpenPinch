Getting Started
===============

This is the shortest supported path from a clean environment to a solved
OpenPinch case. The only currently compatibility-protected Python entry point
is :func:`OpenPinch.main.pinch_analysis_service`.

Install
-------

Install the base package for validation and targeting:

.. code-block:: bash

   python -m pip install openpinch

Optional extras provide notebook, dashboard, Brayton-cycle, and HEN-synthesis
dependencies. They do not expand the compatibility-protected Python API.

.. list-table::
   :header-rows: 1

   * - Extra
     - Use when you need
     - Command
   * - notebook
     - Jupyter, Plotly, or Excel tooling
     - ``python -m pip install "openpinch[notebook]"``
   * - dashboard
     - Streamlit dashboard review
     - ``python -m pip install "openpinch[dashboard]"``
   * - synthesis
     - solver-backed HEN development
     - ``python -m pip install "openpinch[synthesis]"`` then ``idaes get-extensions``
   * - brayton_cycle
     - TESPy-backed Brayton-cycle development
     - ``python -m pip install "openpinch[brayton_cycle]"``

OpenPinch currently targets Python 3.14.

Run the First Solve
-------------------

.. code-block:: python

   from OpenPinch.main import pinch_analysis_service

   result = pinch_analysis_service(
       {
           "streams": [
               {
                   "name": "Hot feed",
                   "zone": "Process",
                   "t_supply": 180.0,
                   "t_target": 80.0,
                   "heat_flow": 1000.0,
               },
               {
                   "name": "Cold feed",
                   "zone": "Process",
                   "t_supply": 20.0,
                   "t_target": 120.0,
                   "heat_flow": 800.0,
               },
           ],
           "utilities": [],
       },
       project_name="first-solve",
   )

   print(result.model_dump(mode="json"))

Read the Result
---------------

The returned model contains ``name``, ``period_id``, ``targets``, ``graphs``,
and ``design``. Inspect each target's hot-utility, cold-utility, and recovered-
heat values through its serialized fields.

Unsupported Advanced Modules
----------------------------

The repository contains concrete owner modules for application workflows,
domain objects, engineering analyses, adapters, optimisation, and
presentation. They support OpenPinch development and the packaged advanced
notebooks, but they are not current external contracts. Deep imports may move
without a compatibility layer.

Use the CLI Only for Notebook Assets
------------------------------------

The CLI copies packaged notebooks; it does not solve cases:

.. code-block:: bash

   openpinch notebook -o notebooks

Next Steps
----------

- :doc:`guides/first-solve-python` for the supported service call.
- :doc:`api/package-root` for the exact compatibility boundary.
- :doc:`developer/architecture` for contributor-facing package ownership.
- :doc:`guides/notebooks-and-sample-cases` for explicitly unsupported advanced
  examples.
