Heat-Recovery ``dt_min``
========================

Purpose
-------

Use the inverse heat-recovery target when the process heat recovery is known
but the corresponding global ``dt_min`` is not. The service answers this
question:

   What is the greatest global ``dt_min`` at which the process can
   still recover the requested heat?

This is a process-targeting calculation over the hot and cold composite
curves. It is not an exchanger minimum approach temperature (EMAT) check. An
EMAT check needs heat-exchanger-network matches and terminal temperatures.

The calculation is non-mutating. It evaluates detached process-stream copies
and does not replace ``problem.results``, populate ``period_results``, change
stream ``delta_t_contribution`` values, or update target-run metadata.

Prerequisites
-------------

- Prepare a :class:`OpenPinch.PinchProblem` with at least one hot and one cold
  process stream for a non-zero recovery opportunity.
- Select a Site, Process Zone, or Unit Operation. Community and Region scopes
  are not direct process-targeting scopes and are rejected with guidance.
  A supplied ``Zone`` object is treated as an address selector: the service
  resolves that address against the current problem and never analyzes streams
  owned by a different problem or workspace case.
- Express the required recovery as a finite, non-negative scalar. Accepted
  forms are Python or NumPy real scalars, ``Decimal``/``Fraction`` values, a
  scalar :class:`OpenPinch.domain.value.Value`, a scalar Pint quantity, or an
  exact ``{"value": number, "unit": "..."}`` mapping. Plain numbers use
  ``INPUT_UNIT_HEAT_FLOW``. Booleans, numeric strings, byte strings,
  sequences, arrays, and arbitrary mappings are rejected instead of being
  coerced.
- Choose a canonical ``period_id`` when the input has multiple operating
  periods.

No optimizer or optional solver dependency is required.

Runnable Workflow
-----------------

Single period
~~~~~~~~~~~~~

The packaged basic case has a zero-``dt_min`` thermodynamic limit of about
5,550 kW. Requesting 4,000 kW gives an equivalent global ``dt_min`` of about
38.75 ``delta_degC``:

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json", project_name="Site")
   ordinary = problem.target.direct_heat_integration(period_id="0")
   cached_results_before = problem.results

   result = problem.target.heat_recovery_dt_min(
       heat_recovery={"value": 4_000.0, "unit": "kW"},
       zone="Site",
       period_id="0",
   )

   result.dt_min
   result.requested_heat_recovery
   result.achieved_heat_recovery
   result.thermodynamic_limit
   result.status

   assert problem.results is cached_results_before
   assert problem.period_results == {}

The returned ``dt_min`` is the full hot-to-cold global spacing. OpenPinch applies
half of each trial ``dt_min`` to every detached hot and cold stream and forces
the detached contribution multiplier to one. Existing heterogeneous
``delta_t_contribution`` values and utilities therefore do not define this
equivalent global ``dt_min``.

Thermodynamic and zero-recovery boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The thermodynamic limit is measured at zero ``dt_min``, but its equivalent
boundary is not necessarily zero. A threshold problem can retain maximum
recovery over a positive global ``dt_min`` interval. Requesting the limit returns
the greatest feasible ``dt_min`` on that plateau with status
``at_thermodynamic_limit``. If recovery decreases immediately, the returned
boundary is zero within the numerical ``dt_min`` tolerance.

A request of zero returns the smallest ``dt_min`` that produces zero recovery,
with status ``zero_recovery_boundary``:

.. code-block:: python

   zero_boundary = problem.target.heat_recovery_dt_min(
       heat_recovery=0.0,
       period_id="0",
   )

   maximum = problem.target.heat_recovery_dt_min(
       heat_recovery=zero_boundary.thermodynamic_limit.model_dump(),
       period_id="0",
   )

   assert zero_boundary.status == "zero_recovery_boundary"
   assert maximum.status == "at_thermodynamic_limit"

The Bleaching process in the packaged pulp-mill case is a threshold example.
Its direct target equals the thermodynamic limit, but the inverse returns a
positive boundary of approximately 58.34505 ``delta_degC``:

.. code-block:: python

   threshold_problem = PinchProblem("pulp_mill.json", project_name="Site")
   direct = threshold_problem.target.direct_heat_integration(
       zone="Bleaching",
       period_id="0",
   )
   threshold = threshold_problem.target.heat_recovery_dt_min(
       heat_recovery=float(direct.heat_recovery_target),
       zone="Bleaching",
       period_id="0",
   )

   assert threshold.status == "at_thermodynamic_limit"
   assert threshold.dt_min.value > 0.0

If one process side is empty, or the hot and cold temperature ranges do not
overlap, the thermodynamic limit can be zero. In that case a zero request is
also at the thermodynamic limit and the ``dt_min`` is zero.

All periods
~~~~~~~~~~~

A scalar is broadcast intentionally. A mapping must contain exactly the
canonical period IDs, and the returned dictionary retains canonical order:

.. code-block:: python

   periods = PinchProblem(
       "crude_preheat_train_multiperiod.json",
       project_name="Crude Train",
   )

   zero_boundaries = (
       periods.target.all_periods.heat_recovery_dt_min(
           heat_recovery=0.0,
       )
   )
   recovery_by_period = {
       period_id: 0.8 * boundary.thermodynamic_limit.value
       for period_id, boundary in zero_boundaries.items()
   }
   period_results = (
       periods.target.all_periods.heat_recovery_dt_min(
           heat_recovery=recovery_by_period,
           workers=2,
       )
   )

``workers`` must be a positive integer. Parallel workers receive isolated
local stream copies and return the same ordered results as ``workers=1``.
Unlike ordinary all-period targeting, this inverse method does not write the
returned values into ``problem.period_results``.

Canonical period keys take precedence over the explicit-unit scalar shape. In
the unusual but valid case where the canonical period IDs are literally
``value`` and ``unit``, ``{"value": ..., "unit": ...}`` is interpreted as the
two-period request. To broadcast an explicitly unit-bearing scalar in that
case, pass a scalar ``Value`` or Pint quantity instead.

Workspace and case batches
~~~~~~~~~~~~~~~~~~~~~~~~~~

The active workspace delegates to its active case. Ordered batches isolate
failures in the normal :class:`OpenPinch.application.workspace.CaseBatchResult`
contract:

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace("basic_pinch.json", project_name="Site")
   workspace.scenario("tight", dt_cont_multiplier=0.8)

   active = workspace.target.heat_recovery_dt_min(
       heat_recovery=4_000.0,
       period_id="0",
   )
   batch = workspace.cases(["baseline", "tight"]).target.heat_recovery_dt_min(
       heat_recovery=4_000.0,
       period_id="0",
   )
   batch_periods = (
       workspace.cases(["baseline", "tight"])
       .target.all_periods.heat_recovery_dt_min(
           heat_recovery=4_000.0,
           workers=2,
       )
   )

   batch.results
   batch.errors
   batch_periods.results

An invalid request in one case appears under ``errors`` without discarding
successful case results.

Expected Output
---------------

The frozen :class:`OpenPinch.contracts.heat_recovery_dt_min.HeatRecoveryDtMinResult`
contains only finite, JSON-serializable values:

.. list-table:: Result fields
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``scope`` and ``period_id``
     - Resolved zone path and canonical operating-period ID.
   * - ``dt_min``
     - Equivalent full global ``dt_min`` with an explicit delta-temperature unit.
   * - ``requested_heat_recovery``
     - Normalized user request with an explicit heat-flow unit.
   * - ``achieved_heat_recovery``
     - Recovery at the returned ``dt_min``. It meets an interior request within
       the documented numerical tolerance.
   * - ``thermodynamic_limit``
     - Maximum process recovery at zero global ``dt_min``.
   * - ``heat_recovery_residual``
     - Achieved recovery minus requested recovery.
   * - ``status``
     - ``solved``, ``at_thermodynamic_limit``, or
       ``zero_recovery_boundary``. A thermodynamic-limit result can carry a
       positive threshold ``dt_min``.
   * - ``iterations``
     - Number of deterministic bisection iterations. A positive threshold
       boundary is solved by bisection; the degenerate zero-limit case returns
       without iterating.

Use the Pydantic serialization surface when recording a study:

.. code-block:: python

   payload = result.model_dump(mode="json")
   json_text = result.model_dump_json()

Heat-flow values honor ``OUTPUT_UNIT_HEAT_FLOW``. The ``dt_min`` honors
``OUTPUT_UNIT_TEMPERATURE`` and is reported as a delta unit, for example
``delta_degC`` or ``delta_degF`` rather than an absolute temperature.
Constructing the result contract directly also validates these dimensions,
non-negative ``dt_min``/recovery/limit values, achieved and requested recovery
against the limit, residual arithmetic, and status-specific relationships.
Numeric strings and Booleans are not accepted as quantity values.

Invalid requests
~~~~~~~~~~~~~~~~

Boolean, negative, non-finite, unsupported-shape, and above-limit requests fail
before an unverified result can be returned. Validation is deliberately strict:
for example, ``"4000"``, ``[4000]``, a zero-dimensional array, and
``{"amount": 4000}`` are not scalar recovery inputs. An above-limit error
reports the requested recovery, calculated limit, scope, period, and units.
All-period mappings with missing or extra period IDs and non-positive
``workers`` values are also rejected.

The numerical search uses a ``1e-6 delta_degC`` ``dt_min`` tolerance, a
``1e-6 kW`` absolute recovery tolerance, a ``1e-9`` relative recovery
tolerance, and a hard 100-iteration limit. Non-finite cascade evaluations,
invalid brackets, inconsistent final bracket re-evaluations, and
non-convergence fail closed. The inverse evaluator preserves exact finite
shifted-temperature levels and strict interval overlap; ordinary targeting
retains its established canonical presentation grid and numerical behavior.

Interpretation
--------------

Recovery is non-increasing as the global ``dt_min`` grows. For an interior
request, OpenPinch returns the greatest feasible ``dt_min`` whose calculated
recovery remains at least the requested value. A positive request at or below
the absolute recovery tolerance is still positive: it returns ``solved`` and
must actually be achieved, rather than being classified as the zero boundary.
This makes flat recovery
plateaus deterministic and gives the most conservative spacing that still
meets the target.

The same greatest-feasible rule applies at maximum recovery. In a threshold
problem it identifies the positive global ``dt_min`` at which external heating
or cooling first becomes necessary. Zero ``dt_min`` describes how the limit is
calculated, not an unconditional inverse answer for that limit.

Use global ``dt_min`` for process-level targeting and early sensitivity studies. Do not use
it as evidence that every proposed exchanger match satisfies EMAT, pressure
drop, area, operability, or control constraints. Carry a promising process
target into a heat-exchanger-network study for those checks.

Notebook 02, ``02_focused_direct_and_total_site.ipynb``, demonstrates the
selected-period inverse beside ordinary direct and Total Site targeting.
Notebook 06, ``06_multiperiod_heat_integration.ipynb``, demonstrates scalar
broadcast and exact period-mapped recovery requests.

Next Steps
----------

- Read :doc:`../fundamentals/pinch-analysis` for global ``dt_min`` and EMAT context.
- Use :doc:`zonal-and-total-site-workflows` to choose an appropriate direct
  process scope.
- See :doc:`../api/pinchproblem` and :doc:`../api/pinchworkspace` for the
  public facade and batch interaction contracts.
- Copy the maintained examples with ``openpinch notebook -o notebooks``.
