Heat-Recovery Approach Temperature
==================================

Purpose
-------

Use the inverse heat-recovery target when the process heat recovery is known
but the corresponding global heat-recovery approach temperature is not. The
service answers this question:

   What is the greatest global HRAT, or delta Tmin, at which the process can
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
- Express the required recovery as a finite, non-negative scalar. Plain
  numbers use ``INPUT_UNIT_HEAT_FLOW``; an explicit value and unit mapping is
  clearer in reusable studies.
- Choose a canonical ``period_id`` when the input has multiple operating
  periods.

No optimizer or optional solver dependency is required.

Runnable Workflow
-----------------

Single period
~~~~~~~~~~~~~

The packaged basic case has a zero-approach thermodynamic limit of about
5,550 kW. Requesting 4,000 kW gives an equivalent global approach of about
38.75 ``delta_degC``:

.. code-block:: python

   from OpenPinch import PinchProblem

   problem = PinchProblem("basic_pinch.json", project_name="Site")
   ordinary = problem.target.direct_heat_integration(period_id="0")
   cached_results_before = problem.results

   result = problem.target.heat_recovery_approach_temperature(
       heat_recovery={"value": 4_000.0, "unit": "kW"},
       zone="Site",
       period_id="0",
   )

   result.approach_temperature
   result.requested_heat_recovery
   result.achieved_heat_recovery
   result.thermodynamic_limit
   result.status

   assert problem.results is cached_results_before
   assert problem.period_results == {}

The returned approach is the full hot-to-cold global spacing. OpenPinch applies
half of each trial approach to every detached hot and cold stream and forces
the detached contribution multiplier to one. Existing heterogeneous
``delta_t_contribution`` values and utilities therefore do not define this
equivalent HRAT.

Thermodynamic and zero-recovery boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Request the zero-approach thermodynamic limit itself to obtain an approach of
zero and status ``at_thermodynamic_limit``. A request of zero instead returns
the smallest approach that produces zero recovery, with status
``zero_recovery_boundary``:

.. code-block:: python

   zero_boundary = problem.target.heat_recovery_approach_temperature(
       heat_recovery=0.0,
       period_id="0",
   )

   maximum = problem.target.heat_recovery_approach_temperature(
       heat_recovery=zero_boundary.thermodynamic_limit.model_dump(),
       period_id="0",
   )

   assert zero_boundary.status == "zero_recovery_boundary"
   assert maximum.status == "at_thermodynamic_limit"

If one process side is empty, or the hot and cold temperature ranges do not
overlap, the thermodynamic limit can be zero. In that case a zero request is
also at the thermodynamic limit and the approach is zero.

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
       periods.target.all_periods.heat_recovery_approach_temperature(
           heat_recovery=0.0,
       )
   )
   recovery_by_period = {
       period_id: 0.8 * boundary.thermodynamic_limit.value
       for period_id, boundary in zero_boundaries.items()
   }
   period_results = (
       periods.target.all_periods.heat_recovery_approach_temperature(
           heat_recovery=recovery_by_period,
           workers=2,
       )
   )

``workers`` must be a positive integer. Parallel workers receive isolated
local stream copies and return the same ordered results as ``workers=1``.
Unlike ordinary all-period targeting, this inverse method does not write the
returned values into ``problem.period_results``.

Workspace and case batches
~~~~~~~~~~~~~~~~~~~~~~~~~~

The active workspace delegates to its active case. Ordered batches isolate
failures in the normal :class:`OpenPinch.application.workspace.CaseBatchResult`
contract:

.. code-block:: python

   from OpenPinch import PinchWorkspace

   workspace = PinchWorkspace("basic_pinch.json", project_name="Site")
   workspace.scenario("tight", dt_cont_multiplier=0.8)

   active = workspace.target.heat_recovery_approach_temperature(
       heat_recovery=4_000.0,
       period_id="0",
   )
   batch = workspace.cases(["baseline", "tight"]).target.heat_recovery_approach_temperature(
       heat_recovery=4_000.0,
       period_id="0",
   )
   batch_periods = (
       workspace.cases(["baseline", "tight"])
       .target.all_periods.heat_recovery_approach_temperature(
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

The frozen :class:`OpenPinch.contracts.heat_recovery.HeatRecoveryApproachResult`
contains only finite, JSON-serializable values:

.. list-table:: Result fields
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``scope`` and ``period_id``
     - Resolved zone path and canonical operating-period ID.
   * - ``approach_temperature``
     - Equivalent full global HRAT with an explicit delta-temperature unit.
   * - ``requested_heat_recovery``
     - Normalized user request with an explicit heat-flow unit.
   * - ``achieved_heat_recovery``
     - Recovery at the returned approach. It meets an interior request within
       the documented numerical tolerance.
   * - ``thermodynamic_limit``
     - Maximum process recovery at zero global approach.
   * - ``heat_recovery_residual``
     - Achieved recovery minus requested recovery.
   * - ``status``
     - ``solved``, ``at_thermodynamic_limit``, or
       ``zero_recovery_boundary``.
   * - ``iterations``
     - Number of deterministic bisection iterations; zero at the
       thermodynamic-limit boundary.

Use the Pydantic serialization surface when recording a study:

.. code-block:: python

   payload = result.model_dump(mode="json")
   json_text = result.model_dump_json()

Heat-flow values honor ``OUTPUT_UNIT_HEAT_FLOW``. The approach honors
``OUTPUT_UNIT_TEMPERATURE`` and is reported as a delta unit, for example
``delta_degC`` or ``delta_degF`` rather than an absolute temperature.

Invalid requests
~~~~~~~~~~~~~~~~

Boolean, negative, non-finite, non-scalar, and above-limit requests fail
before an unverified result can be returned. An above-limit error reports the
requested recovery, calculated limit, scope, period, and units. All-period
mappings with missing or extra period IDs and non-positive ``workers`` values
are also rejected.

The numerical search uses a ``1e-6 delta_degC`` approach tolerance, a
``1e-6 kW`` absolute recovery tolerance, a ``1e-9`` relative recovery
tolerance, and a hard 100-iteration limit. Non-finite cascade evaluations,
invalid brackets, and non-convergence fail closed.

Interpretation
--------------

Recovery is non-increasing as the global approach grows. For an interior
request, OpenPinch returns the greatest feasible approach whose calculated
recovery remains at least the requested value. This makes flat recovery
plateaus deterministic and gives the most conservative spacing that still
meets the target.

Use HRAT for process-level targeting and early sensitivity studies. Do not use
it as evidence that every proposed exchanger match satisfies EMAT, pressure
drop, area, operability, or control constraints. Carry a promising process
target into a heat-exchanger-network study for those checks.

Notebook 02, ``02_focused_direct_and_total_site.ipynb``, demonstrates the
selected-period inverse beside ordinary direct and Total Site targeting.
Notebook 06, ``06_multiperiod_heat_integration.ipynb``, demonstrates scalar
broadcast and exact period-mapped recovery requests.

Next Steps
----------

- Read :doc:`../fundamentals/pinch-analysis` for HRAT and EMAT context.
- Use :doc:`zonal-and-total-site-workflows` to choose an appropriate direct
  process scope.
- See :doc:`../api/pinchproblem` and :doc:`../api/pinchworkspace` for the
  public facade and batch interaction contracts.
- Copy the maintained examples with ``openpinch notebook -o notebooks``.
