# Domain Entities

- `StreamID`: existing endpoint classification enum; only Process and Utility
  are valid on heat exchangers.
- `HeatExchanger`: runtime HEN link whose role fields migrate to StreamID.
- `HeatExchangerNetwork`: runtime ordered network and canonical JSON source.
- `HeatExchangerAreaSliceSchema`, `HeatExchangerPeriodStateSchema`,
  `HeatExchangerSchema`, and `HeatExchangerNetworkSchema`: transport-only
  Pydantic records owned by the input contract.
- `TargetInput`: top-level request schema with an optional transport network.
