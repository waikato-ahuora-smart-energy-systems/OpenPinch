Entrypoints and Assets
======================

This page documents the repository-level entrypoints and packaged assets that
support the main OpenPinch workflows outside the core analysis modules.

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

These modules back the public ``problem.design`` HEN synthesis accessors. The
entry module owns dispatch and problem result caching; targeting service modules
own method-specific orchestration.

.. automodule:: OpenPinch.services.heat_exchanger_network_synthesis.heat_exchanger_network_synthesis_entry
   :members:

.. automodule:: OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.open_hens_method
   :members:

.. automodule:: OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.pinch_design_method
   :members:

.. automodule:: OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.thermal_derivative_method
   :members:

.. automodule:: OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.network_evolution_method
   :members:

.. automodule:: OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.topology
   :members:

.. automodule:: OpenPinch.services.heat_exchanger_network_synthesis.common.execution.pathways
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
