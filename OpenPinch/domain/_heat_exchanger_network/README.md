# Heat Exchanger Network Helpers

## Overview
Private package owned by `HeatExchangerNetwork` in `OpenPinch.domain.heat_exchanger_network`. It currently holds one typing alias for labelled access to network metrics. No module in the package imports it.

## Interface
- `HeatExchangerNetworkLabelKey` (`types.py`): type alias `str | HeatExchangerNetworkLabel` for a label argument.

## Invariants
None.

## Side Effects
None.

## Verification
- Run: `uv run --no-sync pytest tests/domain/test_heat_exchanger_network.py`
