Pinch Analysis
==============

Pinch analysis is the thermodynamic basis for the core OpenPinch workflow. The
main objective is to determine how much heat can be recovered internally before
external heating or cooling utilities are required.

OpenPinch uses pinch analysis to answer three primary questions:

- What is the minimum hot utility target?
- What is the minimum cold utility target?
- Where is the pinch-constrained temperature region?

Thermal Framing
---------------

In OpenPinch terms:

- hot process streams release heat as they cool
- cold process streams require heat as they warm
- utilities satisfy whatever part of the thermal load cannot be recovered
  internally

The package reports this through summary metrics such as:

- `Hot Utility Target`
- `Cold Utility Target`
- `Heat Recovery`
- `Hot Pinch`
- `Cold Pinch`

Analysis Dataflow
-----------------

The core OpenPinch solve path can be read as one analysis-dataflow diagram:

.. code-block:: text

   input files / schemas
       -> validated streams, utilities, and options
       -> prepared Zone hierarchy
       -> direct and/or indirect targeting services
       -> target models attached to zones
       -> TargetOutput summaries and graph data
       -> tables, plots, Excel export, and dashboard views

This matters because the same prepared model feeds the ``PinchProblem``
wrapper, the service layer, and the packaged notebooks. The CLI only copies
notebook sources; it does not run this solve path.

Minimum Approach Temperature
----------------------------

OpenPinch uses ``delta_t_contribution`` as the runtime stream temperature-approach
assumption for streams and many utility calculations.

Conceptually:

- a larger ``delta_t_contribution`` makes heat recovery more conservative
- a smaller ``delta_t_contribution`` increases the apparent recovery potential
- the pinch location and utility targets depend on this assumption

The package exposes base ``delta_t_contribution`` and
``effective_delta_t_contribution`` values on runtime streams,
so zone-level multiplier studies can alter the effective shift while preserving
the original input.

Global ``dt_min`` and exchanger EMAT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process-level global ``dt_min`` (sometimes called HRAT) is the separation
between the shifted hot and cold process composite curves. OpenPinch can invert
that relationship: given a required process heat recovery, it calculates the
equivalent global ``dt_min`` with
``problem.target.heat_recovery_dt_min(...)``. The calculation
uses uniform half-shifts on detached hot and cold process streams, so existing
stream-specific ``delta_t_contribution`` values and utilities do not affect the
answer.

At zero global ``dt_min``, the process cascade gives the thermodynamic limit:
the largest recovery available without allowing the shifted composites to
cross. Increasing the global ``dt_min`` cannot increase process heat recovery.
The inverse service searches this monotonic relationship between zero and the
finite no-overlap spacing.

The inverse can have more than one temperature answer when recovery is flat
over a ``dt_min`` interval. For an interior request, OpenPinch returns the
greatest feasible ``dt_min`` that still meets the requested recovery. For a zero
request it returns the smallest zero-recovery boundary. The same
greatest-feasible rule applies at the thermodynamic limit.

This distinction matters for a threshold problem, where no external heating or
no external cooling is required. Maximum recovery can persist over a positive
global ``dt_min`` plateau. Its thermodynamic-limit inverse is the greatest
``dt_min`` on that plateau, not automatically zero. A non-threshold problem whose
recovery decreases immediately has a limit boundary at approximately zero.
These rules make all boundary and plateau results deterministic.

Global ``dt_min`` is not exchanger minimum approach temperature (EMAT). EMAT constrains an
individual exchanger and requires HEN match and temperature data. An inverse
process target cannot establish exchanger feasibility by itself. This
distinction follows established pinch-analysis terminology in the `process
integration user guide
<https://moodle.unige.ch/pluginfile.php/386097/mod_folder/content/0/Pinch_Analysis_and_Process_Integration.pdf>`_.

See :doc:`../guides/heat-recovery-dt-min` for runnable
single-period, multiperiod, workspace, and batch workflows.

What The Pinch Represents
-------------------------

The pinch is the temperature region where the process is most constrained with
respect to heat recovery under the chosen temperature-approach assumptions.

Practically, this means:

- utility targets are determined by the interval cascade through this
  constrained region
- graph interpretation often starts here when a case is difficult to improve
- direct and indirect integration workflows both depend on the same broad idea,
  but they apply it at different system scopes

What OpenPinch Adds Beyond The Textbook Core
--------------------------------------------

Classical pinch targeting is the starting point, not the whole package.
OpenPinch extends the workflow with:

- hierarchical zone modeling
- indirect / site-level targeting
- multiple graph views
- optional Heat Pump and refrigeration workflows
- optional turbine cogeneration post-processing
- programmatic and file-backed workflows over the same core engine

Recommended Follow-On Pages
---------------------------

- :doc:`problem-table-and-temperature-shifting`
- :doc:`direct-vs-indirect-integration`
- :doc:`../guides/heat-recovery-dt-min`
- :doc:`graphs-and-interpretation`
