Configuration Options
=====================

Every key accepted in ``TargetInput.options``, a per-call ``options``
argument, ``update_options(...)`` or a workspace scenario, with its type,
default, unit, limits and meaning. The tables are generated from
``configuration_options()`` (see :doc:`schemas-and-config`), so they match the
installed version.

Setting Options
---------------

.. code-block:: python

   problem.update_options({"THERMAL_DT_CONT": 10.0, "HPR_AMBIENT_AIR_ENABLED": False})
   problem.target.direct_heat_integration(options={"OUTPUT_UNIT_HEAT_FLOW": "MW"})

Effective values resolve as ``named keyword > options > stored config >
default``. A per-call ``options`` argument applies to that call only;
``update_options(...)`` changes the stored value for later calls.

The ``problem`` and ``input_units`` groups, ``THERMAL_DT_CONT``,
``THERMAL_DT_PHASE_CHANGE``, ``THERMAL_HTC`` and ``COSTING_UTILITY_PRICE`` are
applied when the problem is loaded. Change them with ``update_options(...)``
or a workspace scenario; a per-call ``options`` argument rejects them with a
``ValueError``.

Method selectors such as ``HPR_TYPE``, ``ORC_MODEL`` and ``POWER_TURB_MODEL``
are not options: the ``problem.target.*`` or ``problem.design.*`` method you
call selects the analysis, and passing a selector raises a ``ValueError``.

Units given in the Unit column are the units of the default. Limits show the
accepted numeric range (``>`` excludes the bound) or the accepted choices.
Options in every group other than Problem, Input Units, Output Units,
Reporting and Thermal are advanced engineering settings.

Options by Group
----------------

.. include:: ../_data/configuration-options.inc
