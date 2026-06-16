Heat Pump and Refrigeration Methods
===================================

OpenPinch treats Heat Pump and refrigeration work as an integration problem
first and a cycle problem second. A candidate cycle is valuable only if its
hot and cold utility streams improve the process heat cascade after the
minimum-approach shifts have been applied.

The public workflows are:

- ``problem.target.direct_heat_pump(...)``
- ``problem.target.indirect_heat_pump(...)``
- ``problem.target.direct_refrigeration(...)``
- ``problem.target.indirect_refrigeration(...)``

These workflows prepare the relevant process or site-level background cascade
and then dispatch one of the configured HPR backends through ``HPR_TYPE``.

Direct and Indirect Placement
-----------------------------

Direct HPR
   Direct methods place the HPR streams inside the selected process or zone
   boundary. They answer questions such as: can a heat pump move heat from this
   zone's available hot side to this zone's heat demand?

Indirect HPR
   Indirect methods place the HPR streams against the aggregated utility
   picture. They are useful for Total Site-style questions where heat may be
   lifted from one zone's surplus into another zone's demand through a utility
   layer.

In both cases, OpenPinch evaluates the HPR as additional hot and cold utility
streams and recomputes process heat cascades. A simplified objective has the
form:

.. math::

   J = \frac{W_\mathrm{HPR} + Q_\mathrm{ext,hot}
       + Q_\mathrm{ext,cold} + P}{Q_\mathrm{target}}

where ``W_HPR`` is electrical work, ``Q_ext,hot`` and ``Q_ext,cold`` are
remaining external utilities after integration, and ``P`` is the feasibility
penalty. The exact accounting may include user-configured heat-to-power and
cold-to-power price ratios in higher-level comparisons, but the physical
interpretation is the same: lower external utility and lower work are better.

Temperature and Duty Conventions
--------------------------------

For heat pumping, the cycle absorbs heat from a source side and rejects useful
heat on a sink side:

.. math::

   Q_\mathrm{cond} = Q_\mathrm{evap} + W

For refrigeration, the useful duty is on the evaporator side:

.. math::

   Q_\mathrm{cond} = Q_\mathrm{cool} + W

OpenPinch uses shifted process temperatures during targeting. The configured
``DT_CONT_HP`` and related approach settings therefore represent integration
temperature margins, not necessarily equipment terminal temperature differences
from a detailed exchanger design.

Optimisation Coordinates and Duty Allocation
--------------------------------------------

HPR optimisation variables are deliberately not final physical duties. The
targeting services decode the optimiser vector into:

- ambient hot/cold duty from ``x_amb``
- condenser and evaporator temperatures from ``x_cond`` and ``x_evap``
- approach settings such as subcooling and internal heat-exchanger temperature
  differences where the selected backend supports them
- total base duty scales such as ``Q_heat_base`` and ``Q_cool_base``
- process-side availability arrays such as ``Q_heat_available`` and
  ``Q_cool_available`` at the candidate stage temperatures

The backend classes then decode bounded split fractions into requested stage
duties:

.. math::

   Q_{\mathrm{request},i}
   = f_i\left(Q_\mathrm{base} - \sum_{j<i} Q_{\mathrm{request},j}\right)

where ``f_i`` is the decoded split fraction for stage ``i``. Requested duties
are clipped to what the process cascade can actually accept or supply:

.. math::

   Q_{\mathrm{model},i}
   = \min(Q_{\mathrm{request},i}, Q_{\mathrm{available},i})

.. math::

   Q_{\mathrm{excess},i}
   = \max(Q_{\mathrm{request},i} - Q_{\mathrm{available},i}, 0)

The physical cycle model receives ``Q_model`` only. Excess duty contributes to
the optimisation penalty instead of being passed into refrigerant or Carnot
cycle calculations. This is important for robustness: a trial optimiser point
may request more heat or cooling than one stage can exchange, but the backend
still solves a physically bounded cycle and reports the over-allocation.

Use the public cycle names exactly as listed in this page.

Cascade Carnot Cycles
---------------------

``HPR_TYPE = "Cascade Carnot cycles"``

This is the fastest and most robust thermodynamic screening method. It does
not solve a refrigerant cycle. Instead, it asks how much heat could be moved
between candidate evaporating and condensing temperature levels if the cycle
performed like a second-law-scaled Carnot device.

For a heat-pump lift from ``T_l`` to ``T_h`` in kelvin, OpenPinch uses:

.. math::

   \mathrm{COP}_h =
   1 + \eta_{\mathrm{II,HPR}}\frac{T_l}{T_h - T_l}

and therefore:

.. math::

   W = \frac{Q_h}{\mathrm{COP}_h}
   \qquad
   Q_l = Q_h - W

When heat recovery or heat-engine behavior is possible between temperature
levels, the heat-engine efficiency is:

.. math::

   \eta_\mathrm{HE} =
   \eta_{\mathrm{II,HE}}\left(1 - \frac{T_l}{T_h}\right)

Distributed duties use an entropic mean temperature:

.. math::

   \bar{T}_S = \frac{\sum_i Q_i}{\sum_i Q_i/T_i}

This method is recommended for early screening, sensitivity studies, and
warm-starting slower simulated-cycle methods.

Parallel Carnot Cycles
----------------------

``HPR_TYPE = "Parallel Carnot cycles"``

This method represents several independent Carnot-like heat-pump stages. Each
stage has one evaporating level and one condensing level. It is less flexible
than the cascade matrix method, but easier to interpret when the user
wants a small number of discrete heat pumps.

For each stage ``k``:

.. math::

   \mathrm{COP}_{h,k} =
   1 + \eta_{\mathrm{II,HPR}}\frac{T_{\mathrm{evap},k}}
   {T_{\mathrm{cond},k} - T_{\mathrm{evap},k}}

.. math::

   W_k = \frac{Q_{\mathrm{cond},k}}{\mathrm{COP}_{h,k}}
   \qquad
   Q_{\mathrm{evap},k} = Q_{\mathrm{cond},k} - W_k

The total work and useful duties are sums over the active stages:

.. math::

   W_\mathrm{tot} = \sum_k W_k
   \qquad
   Q_\mathrm{cond,tot} = \sum_k Q_{\mathrm{cond},k}

Use this method when you want a simple, staged conceptual target without
committing to refrigerant properties.

Internally this method is solved by the
``ParallelCarnotCycles`` backend class. It receives a total heat base duty,
stage heat split fractions, and process availability arrays. The backend
allocates per-stage ``Q_cond`` values before applying COP, heat-engine, or heat
recovery calculations.

Brayton Cycle
-------------

``HPR_TYPE = "Brayton cycle"``

The Brayton backend represents a gas-cycle heat pump. It is useful where
sensible heat profiles and larger temperature glides are more important than
phase-change behavior. In a simple idealized Brayton heat pump:

.. math::

   \frac{T_2}{T_1} =
   \left(\frac{p_2}{p_1}\right)^{(\gamma - 1)/\gamma}

with compressor efficiency applied as:

.. math::

   h_{2,\mathrm{actual}} =
   h_1 + \frac{h_{2,s} - h_1}{\eta_\mathrm{comp}}

The cycle rejects heat on the high-pressure side and absorbs heat on the
low-pressure side:

.. math::

   Q_h = \dot{m}(h_2 - h_3)
   \qquad
   Q_l = \dot{m}(h_1 - h_4)
   \qquad
   W = Q_h - Q_l

The implementation uses the TESPy-backed Brayton unit model when the optional
Brayton dependency is installed. Install ``openpinch[brayton_cycle]`` for this
workflow.

Parallel Vapour-Compression Cycles
----------------------------------

``HPR_TYPE = "Parallel vapour compression cycles"``

This backend solves one or more independent vapour-compression cycles using
CoolProp fluid properties. It is slower than Carnot targeting but captures
refrigerant-specific saturation pressure, compressor discharge, subcooling,
superheating, and internal heat-exchanger effects.

For a stage with mass flow ``m_dot``:

.. math::

   q_\mathrm{evap} = h_0 - h_3
   \qquad
   q_\mathrm{cond} = h_1 - h_2
   \qquad
   w = q_\mathrm{cond} - q_\mathrm{evap}

.. math::

   Q_\mathrm{evap} = \dot{m}q_\mathrm{evap}
   \qquad
   Q_\mathrm{cond} = \dot{m}q_\mathrm{cond}
   \qquad
   W = \dot{m}w

The actual compressor outlet is calculated from an isentropic outlet enthalpy:

.. math::

   h_{1,\mathrm{actual}} =
   h_0 + \frac{h_{1,s} - h_0}{\eta_\mathrm{comp}}

Use this backend when refrigerant selection or realistic compressor work
matters and independent stages are a reasonable process model.

The simulated backend follows the same base/split convention as the Carnot
backends. The optimiser controls ``x_heat_base`` plus ``x_heat_split`` for heat
pump solves, or ``x_cool_base`` plus ``x_cool_split`` for refrigeration solves.
The aggregate backend clips requested duties to process availability before it
calls the child ``VapourCompressionCycle`` unit models. Child cycles therefore
receive concrete physical ``Q_heat`` or ``Q_cool`` duties, not optimiser
fractions.

Cascade Vapour-Compression Cycles
---------------------------------

``HPR_TYPE = "Cascade vapour compression cycles"``

The cascade vapour-compression backend couples several refrigerant cycles
through cascade heat exchangers. A lower-temperature stage rejects part of its
condenser heat internally to the evaporator side of the next higher-temperature
stage. External streams exclude the internal cascade heat by default so the
process utility accounting only sees source-side heat, direct useful heat, and
work.

For adjacent stages ``i`` and ``i+1``:

.. math::

   Q_{\mathrm{cas},i} =
   Q_{\mathrm{evap},i+1}

subject to a cascade approach constraint:

.. math::

   T_{\mathrm{cond},i,\mathrm{boundary}}
   - T_{\mathrm{evap},i+1}
   \geq \Delta T_\mathrm{cascade}

The total useful external heat is:

.. math::

   Q_{\mathrm{heat,external}} =
   \sum_i Q_{\mathrm{cond},i}
   - \sum_i Q_{\mathrm{cas},i}

and total work is:

.. math::

   W_\mathrm{tot} = \sum_i W_i

Use this backend when one refrigerant lift is too large or when a staged
refrigerant system is physically more plausible than independent cycles.

The cascade backend also owns base/split duty allocation. For heat-pump solves,
condenser-side split fractions allocate the useful heating pool and the
evaporator-side split fractions allocate source duty for all explicitly sized
source stages. The final cascade source stage can still be inferred from the
coupled heat balance when appropriate.

Vapour Compression with MVR Cascade
-----------------------------------

``HPR_TYPE = "Vapour compression with MVR cascade"``

This backend is heat-pump-only in the current implementation. It couples a
vapour-compression low stage with a serial mechanical vapour recompression
high stage. The hottest VC condenser segment generates vapour for the first
MVR stage. Each MVR stage compresses the vapour to a higher saturation
temperature, rejects useful process heat, and passes the uncondensed vapour to
the next MVR stage.

The first MVR evaporating temperature is derived from the solved VC condenser
cascade boundary:

.. math::

   T_{\mathrm{evap,MVR},1} =
   T_{\mathrm{VC,boundary}} - \Delta T_\mathrm{cascade}

Each MVR stage is constrained to a bounded saturation lift:

.. math::

   0 < \Delta T_{\mathrm{lift},j} \leq 20\ \mathrm{K}

.. math::

   T_{\mathrm{cond,MVR},j} =
   T_{\mathrm{evap,MVR},j} + \Delta T_{\mathrm{lift},j}

and the next stage receives vapour at the previous stage's condensing
saturation temperature:

.. math::

   T_{\mathrm{evap,MVR},j+1} = T_{\mathrm{cond,MVR},j}

The first stage source-vapour mass flow is set by the internal VC heat used as
an MVR source duty:

.. math::

   \dot{m}_{\mathrm{src},1} =
   \frac{Q_\mathrm{src}}{h_{0,1} - h_{\mathrm{liq},1}}

Each MVR stage is modelled as dry source-vapour compression followed by a
separate post-compression internal liquid-injection desuperheating step. The
compressor work is therefore based on the source-vapour mass flow before
injection; the injected liquid increases the vapour mass available after the
desuperheating step. The dry compressor discharge is:

.. math::

   h_{1,\mathrm{actual}} =
   h_0 + \frac{h_{1,s} - h_0}{\eta_\mathrm{MVR}}

.. math::

   W_j =
   \frac{\dot{m}_{\mathrm{src},j}(h_{1,\mathrm{actual}} - h_0)}
   {\eta_\mathrm{motor}}

Injected liquid from the condenser outlet consumes the dry discharge superheat
internally. This is an idealized post-compression desuperheater model, not a
wet-compression suction spray model. With saturated vapour after injection:

.. math::

   r_{\mathrm{inj},j} =
   \frac{h_{1,\mathrm{actual}} - h_{\mathrm{vap},j}^\mathrm{sat}}
   {h_{\mathrm{vap},j}^\mathrm{sat} - h_{\mathrm{liq,out},j}}

.. math::

   \dot{m}_{\mathrm{vap},j} =
   \dot{m}_{\mathrm{src},j}(1 + r_{\mathrm{inj},j})

The injected liquid is represented as an ideal internal/recycle liquid stream
at the condenser outlet state. The v1 targeting model updates the downstream
vapour mass flow and heat balance, but it does not add a separate condensate
availability constraint requiring the process-condensed fraction to supply all
injection liquid. Treat this as a targeting approximation rather than a
detailed separator and recycle-loop design.

The useful MVR process heat is therefore condensation and subcooling of the
process split fraction ``f_j`` of the post-injection vapour mass:

.. math::

   Q_{\mathrm{heat,MVR},j} =
   \dot{m}_{\mathrm{vap},j} f_j
   (h_{\mathrm{vap},j}^\mathrm{sat} - h_{\mathrm{liq,out},j})

The remaining vapour continues to the next stage:

.. math::

   \dot{m}_{\mathrm{src},j+1} =
   \dot{m}_{\mathrm{vap},j}(1 - f_j)

The VC stage heat allocation uses the same ``Q_heat_base`` and
``x_heat_split`` convention as the other heat-pump backends. MVR-specific
fractions are named as splits as well: ``mvr_source_split`` describes how much
of the selected VC condenser heat becomes source vapour for the MVR train, and
``x_mvr_process_split`` controls the process-condensed fraction at each MVR
stage.

The final stage condenses the remaining vapour. The external stream collection
includes VC source-side heat, VC direct process heat, and MVR process heat
profiles. It excludes both the internal VC-to-MVR source heat and the internal
liquid-injection desuperheating heat unless explicitly requested by lower-level
model code.

Direct Process Gas/Vapour MVR Components
----------------------------------------

``problem.add_component.process_mvr(...)``

Direct process MVR is a process-component workflow rather than an HPR target
backend. It starts from a prepared ``PinchProblem`` case, selects one or more
hot gas/vapour process streams, solves a direct mechanical-vapour-recompression
replacement profile, and swaps the original streams out of the active hot
stream collection. The original streams remain attached to the model with
``active=False`` while the generated replacement streams become active.

The component can be configured by stage count, saturation-temperature lift or
pressure ratio, compressor efficiency, motor efficiency, and whether liquid
injection is represented as an internal desuperheating step. After the
component is active, ordinary direct and indirect targeting routines consume
the mutated stream collection and report the added process-component work in
the solved target summaries.

This workflow is useful when the engineering question is not "where should a
new HPR cycle be placed?" but "what happens to the integration target if this
existing vapour stream is recompressed?" Use ``PinchWorkspace`` copies when you
need baseline, dry MVR, and liquid-injection MVR cases side by side.

Choosing a Method
-----------------

Use the simplest backend that answers the engineering question:

.. list-table::
   :header-rows: 1

   * - Question
     - Suggested backend
   * - Early screening, many possible temperature placements
     - Cascade Carnot cycles
   * - A few easy-to-explain conceptual stages
     - Parallel Carnot cycles
   * - Refrigerant-specific independent cycles
     - Parallel vapour compression cycles
   * - Large lift split over refrigerant cascade stages
     - Cascade vapour compression cycles
   * - Sensible gas-cycle heat pump with temperature glide
     - Brayton cycle
   * - VC low stage feeding a serial recompression train
     - Vapour compression with MVR cascade
   * - Existing vapour stream recompressed before targeting
     - Direct process gas/vapour MVR component

Recommended Follow-On Pages
---------------------------

- :doc:`../guides/heat-pump-workflows`
- :doc:`../examples/notebook-series`
- :doc:`../reference/api-heat-pump`
- :doc:`../api/service-layer`
