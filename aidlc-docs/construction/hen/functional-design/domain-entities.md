# Domain Entities

- Parent solver axes retain existing hot/cold stream identities.
- Segment profile tensors store padded ordered breakpoints, duties, temperatures, HTCs, masks, and identities.
- Cumulative heat coordinates map parent stage state to temperature.
- Internal frozen `HeatExchangerAreaSlice` records represent local segment-pair
  calculations nested under one parent `HeatExchanger`; the parent exposes
  period duty/area totals and maximum period-total design area.
