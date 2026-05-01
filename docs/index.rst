OpenPinch Toolkit
=======================

OpenPinch provides advanced Pinch Analysis and Total Site Integration tooling
in Python. These pages describe the core concepts, how to run analyses, how to
use the packaged notebooks and sample cases, and the public API.

Source code: GitHub_

Pinch Analysis
--------------
Pinch analysis is a systematic framework for minimising energy consumption in
industrial processes by matching heat sources and sinks to maximise heat
recovery. It identifies the "pinch" temperature where the process is most
constrained, guiding engineers to redesign heat exchanger networks, set
realistic utility targets, and prioritise retrofits that deliver the largest
energy efficiency gains.

Total Site Integration
----------------------
Total site integration extends pinch principles across an entire industrial
complex, coordinating the heat recovery potential of multiple process units,
utilities, and energy storage systems. By viewing the site as a single thermal
ecosystem, engineers can balance steam levels, share surplus heat between
plants, and cut overall fuel demand without compromising production targets.
OpenPinch provides the tools to identify cross-unit heat matches, quantify
utility interactions, and prioritize integration projects that deliver the
largest system-wide savings. A key difference between pinch analysis and 
total site integration is the method of heat recovery: direct (process-to-process)
versus indirect (process-to-utility-to-process).


Highlights
----------
- Multi-scale targeting for process, site, and regional studies
- Multiple utility targeting (isothermal and non-isothermal) with assisted heat
  integration options
- Composite-curve and grand-composite-curve graph generation
- Imports established Excel data templates and exports detailed reports
- Packaged sample cases and notebook workflows for first-time users
- Pydantic schema models for validated programmatic workflows


History of OpenPinch
--------------------
OpenPinch started as part of Tim Walmsley's PhD research at the University of
Waikato and has continued evolving through industrial projects and community
contributions. This implementation brings the capabilities of the long-running 
Excel/VBA workbook into a modern Python API so engineers can automate targeting 
studies, integrate with other software tools and projects, and embed results 
into wider optimisation workflows.

The packaged user workflow is now organized around:

- ``openpinch sample`` and ``openpinch run`` for the first solve
- ``openpinch graph`` for graph export
- ``openpinch notebook`` for guided notebook workflows
- :class:`OpenPinch.PinchProblem` for Python and notebook usage

.. _GitHub: https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch/

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   user-guide/quickstart
   user-guide/notebooks
   reference/index
