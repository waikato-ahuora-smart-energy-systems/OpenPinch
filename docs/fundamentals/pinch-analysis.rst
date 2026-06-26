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

This matters because the same prepared model feeds the CLI, the
``PinchProblem`` wrapper, the service layer, and the packaged notebooks.

Minimum Approach Temperature
----------------------------

OpenPinch uses `dt_cont` as the main continuous temperature-approach
assumption for streams and many utility calculations.

Conceptually:

- a larger `dt_cont` makes heat recovery more conservative
- a smaller `dt_cont` increases the apparent recovery potential
- the pinch location and utility targets depend on this assumption

The package supports both base and active `dt_cont` values on runtime streams,
so zone-level multiplier studies can alter the effective shift while preserving
the original input.

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
- :doc:`graphs-and-interpretation`
