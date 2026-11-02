Application Entrypoints and Resources
=====================================

Application workflows begin at the package root:

.. code-block:: python

   from OpenPinch import PinchProblem, PinchWorkspace

The ``openpinch notebook`` CLI subcommand copies maintained tutorial sources,
and ``openpinch schema`` writes the JSON Schema of each public contract. Resource
helpers under :mod:`OpenPinch.resources` list, inspect, read, and copy packaged
sample cases and notebooks.

Advanced contributor modules are documented in :doc:`api-analysis` and the
generated module index.
