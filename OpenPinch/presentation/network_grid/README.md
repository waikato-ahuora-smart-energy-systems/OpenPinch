# Heat exchanger network grid

## Overview
Builds grid diagrams of heat exchanger networks and the plain-data design view of a synthesis result. `geometry.py` normalises a network into a grid model for one period; `layout.py` and `view.py` turn that model into `contracts` objects without a plotting backend; `service.py` and `rendering.py` draw it as a Plotly figure through a small matplotlib-like axes shim (`plotly.py`). Plotly is optional (extra `synthesis`).

## Interface
- `build_grid_diagram` (`service.py`): grid diagram for one `HeatExchangerNetwork`, or for a sequence (all, or the one selected by `index`); takes `period_id` and `temperature_scaled`.
- `HeatExchangerNetworkGridDiagram`, `build_result_grid_diagram` (`results.py`): dataclass holding `fig`, `ax`, `network` and `grid_model`, with `show` and `save` (HTML by suffix, otherwise `write_image`); builds the diagram of a ranked synthesis network.
- `build_network_view` (`view.py`): ranking table, totals, utility duties and grid of one ranked network as `HeatExchangerNetworkView`.
- `build_grid_layout` (`layout.py`): the grid model as a `HeatExchangerNetworkGridLayout` contract.
- `build_grid_model`, `neighboring_x_positions` (`geometry.py`): normalised topology of a network in one period; nearest coordinates either side of a value.
- `GridDiagramMatch`, `HeatExchangerNetworkGridModel` (`state.py`): frozen dataclasses for one match and for the whole topology.
- `require_network_grid_diagram_dependency`, `MissingNetworkGridDiagramDependencyError`, `GRAPHING_EXTRA` (`dependencies.py`): import an optional module or raise an actionable `ImportError`.
- Renderer internals: `_PlotlyGridRenderer` (`rendering.py`); label and tooltip helpers `add_label`, `add_duty_labels`, `configure_figure`, `exchanger_tooltip`, `short_stream_name` (`labels.py`); temperature helpers `stream_endpoint_temperatures`, `temperature_scale`, `x_from_temperature` and label functions (`temperatures.py`); `_PlotlyAxes`, `_PlotlyLine`, `_PlotlyText`, `_SplitGroup` (`plotly.py`); layout constants (`_constants.py`).
- Subdirectories: none.

## Invariants
- `period_id` is required for a network with more than one period and must be one of its periods; an unknown or missing id, or, with no `period_id`, an empty-period network, raises `ValueError`. [geometry.py:build_grid_model]
- A match is drawn only if it is active, `match_allowed` and its duty exceeds `1e-5` of the largest active duty in that period. [geometry.py:_is_significant_match] [_constants.py:_DUTY_RELATIVE_TOLERANCE]
- Exchangers with the same stream pair and stage are merged, adding duty and areas; the area is `None` if any part has none. [geometry.py:_add_or_accumulate] [geometry.py:_summed_area]
- Recovery exchangers without a stage add streams but are not drawn as matches. [geometry.py:build_grid_model]
- Stages are `1..stage_count` when the network has a stage count, else the stages seen, else `(1,)`. [geometry.py:build_grid_model]
- Hot and cold streams keep first-seen order; recovery matches are sorted by stage, then hot, then cold index; hot-utility matches by cold-stream index and cold-utility matches by hot-stream index. [geometry.py:build_grid_model]
- `branch_counts` hold only (stream, stage) pairs with more than one recovery match. [geometry.py:build_grid_model]
- Temperatures are absolute kelvin and shown as degC by subtracting 273.15; duties are in kW and shown in MW. [temperatures.py:add_temperature_labels] [labels.py:add_duty_labels]
- `build_grid_diagram` rejects a non-network, a non-sequence, an empty sequence, a non-network item (`TypeError`/`ValueError`) and an out-of-range `index` (`IndexError`). [service.py:build_grid_diagram]
- Axis labels in the figure are `H1..`, `C1..` by position, not the stream names. [labels.py:configure_figure]
- Layering: presentation may import only `adapters`, `analysis`, `contracts`, `domain` and `presentation`. [tests/architecture/test_dependency_rules.py:LAYER_ALLOWED_ROOTS]

## Side Effects
- Imports Plotly lazily. [service.py:_build_single_grid_diagram] [dependencies.py:require_network_grid_diagram_dependency]
- `show` opens a viewer; `save` writes HTML or an image file; without Kaleido an image save raises `MissingSynthesisDependencyError` (extra `synthesis`) and any other export error from Plotly is left as it is. [results.py:HeatExchangerNetworkGridDiagram]
- Renderer objects hold mutable per-diagram state (label slot counts, tooltips, split groups) and add traces and annotations to their figure. [rendering.py:_PlotlyGridRenderer] [labels.py:add_label]

## Verification
- Run: `uv run --no-sync pytest tests/application/test_hen_design_view_data.py tests/analysis/heat_exchanger_networks/test_reporting.py` (no test directory targets this package alone; these two cover it).
