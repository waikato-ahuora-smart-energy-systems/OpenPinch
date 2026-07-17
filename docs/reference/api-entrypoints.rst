Entrypoints and Assets
======================

This page documents repository tooling and packaged assets. The sole protected
Python contract is :func:`OpenPinch.main.pinch_analysis_service`.

CLI and Packaged Resources
--------------------------

.. automodule:: OpenPinch.__main__
   :members:

.. automodule:: OpenPinch.resources
   :members:

Packaged Data Assets
--------------------

.. automodule:: OpenPinch.data
   :no-members:

.. automodule:: OpenPinch.data.sample_cases
   :no-members:

.. automodule:: OpenPinch.data.notebooks
   :no-members:

Heat Exchanger Network Synthesis
--------------------------------

These modules back the internal ``problem.design`` HEN synthesis accessors. The
entry module owns dispatch and problem result caching; targeting service modules
own method-specific orchestration.

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.service
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.targeting.thermal_derivative_method
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.targeting.network_evolution_method
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.targeting.topology
   :members:

.. automodule:: OpenPinch.analysis.heat_exchanger_networks.execution.pathways
   :members:

Streamlit App
-------------

.. automodule:: streamlit_app
   :members:

Repository Maintenance Scripts
------------------------------

.. automodule:: scripts.build_docs
   :members:

.. automodule:: scripts.build_dist
   :members:

.. automodule:: scripts.format_repo
   :members:

.. automodule:: scripts.optional_install_smoke
   :members:

.. automodule:: scripts.update_toolchain
   :members:
