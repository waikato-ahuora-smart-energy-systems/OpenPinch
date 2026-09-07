# Numerical entities

Frozen HPRLoadSummary records available, selected and achieved service, cycle
heating/cooling, work, ambient exchanges and units. HPRResidualProfile stores
temperature, normalized net cascade and separated heating/cooling demands on
one grid. HPRResidualData adds selected period, mode and physical boundary
information. Application provenance will be attached by unit 3 without mutable
backend references. Fields on HeatPumpTargetBase are optional for unsupported
weighted legacy targets; conversion requires populated scalar data.

PBT-01 compliant: categories and oracles identified in business-logic-model.md.
Other PBT verification applies during implementation. Security/Resiliency disabled.
