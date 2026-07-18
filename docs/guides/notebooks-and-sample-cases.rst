Notebooks and Sample Cases
==========================

List packaged assets:

.. code-block:: python

   from OpenPinch.resources import (
       list_notebooks,
       list_sample_cases,
       notebook_metadata,
       sample_case_metadata,
   )

   print(list_sample_cases())
   print(list_notebooks())

Copy all tutorials:

.. code-block:: bash

   openpinch notebook -o notebooks

Copy one tutorial:

.. code-block:: bash

   openpinch notebook --name 01_first_solve_and_core_curves.ipynb -o notebooks

The eighteen tutorials are described in :doc:`../examples/notebook-series`.
Their complete public-operation ownership is published in
:doc:`../examples/tutorial-coverage-map`.
