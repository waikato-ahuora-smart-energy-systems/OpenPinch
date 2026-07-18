Heat Pump and Refrigeration Methods
===================================

OpenPinch separates engineering model choice into descriptive methods:

``carnot_heat_pump`` and ``carnot_refrigeration``
   Fast thermodynamic screening. Use utility-placement and cascade-topology
   booleans to express the physical arrangement.

``vapour_compression_heat_pump`` and ``vapour_compression_refrigeration``
   Refrigerant-specific simulated cycles with explicit refrigerant candidates.

``brayton_heat_pump`` and ``brayton_refrigeration``
   Brayton-cycle studies using the model-specific backend.

``mvr_heat_pump``
   Vapour-compression/MVR cascade targeting.

The method name selects the model; configuration does not. Named
``load_fraction``, ``load_duty``, or ``period_loads`` values avoid a separate
load-mode string. Only one load form may be supplied per call.

Placement and Topology
----------------------

Direct placement couples the HPR model to a process heat-integration target.
Utility placement couples it to the site utility system. For Carnot methods,
``is_utility_heat_pump`` or ``is_utility_refrigeration`` expresses that choice.

Parallel topology represents independent lifts. Cascade topology connects
successive temperature lifts and is selected with ``is_cascade_cycle`` where
the model supports it. MVR is a separate named workflow rather than a hidden
cycle-string combination.

Interpretation
--------------

Compare candidate results using utility reduction, recovered heat, COP, work,
temperature lift, and annualized cost. A thermodynamically feasible candidate
is not automatically the lowest-cost retrofit; interpret HPR targets alongside
the Grand Composite Curve and net-load profiles.

Simulated-cycle integration accounting retains the model-specific compressor,
expander, heat-exchanger, operating-cost, and annualized-capital contributions
used by result summaries. Multiperiod summaries weight operating quantities and
size shared equipment against the governing period rather than averaging away
the peak design requirement.

See :doc:`../guides/heat-pump-workflows` and notebooks 08 through 11 in
:doc:`../examples/notebook-series`.
