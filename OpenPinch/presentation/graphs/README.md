# Graph selection and rendering

## Overview
Turns the serialised graph data of a solved problem into Plotly figures and HTML files. `plotly.py` converts one graph mapping to a figure, `problem.py` holds the accessor behind `problem.plot` (inventory, selection, per-graph methods, HTML export), and `simple.py` has small debugging plots. Graph data comes from `analysis.graphs.service`; Plotly is an optional dependency.

## Interface
- `build_plotly_figure` (`plotly.py`): builds a figure from one graph mapping with a `segments` list of coloured line segments.
- `_PlotAccessor` (`problem.py`): constructed with a problem and an optional heat-pump/refrigeration graph selector; the `PinchProblem` plot accessor wraps it. Methods:
  - `catalog`, `data`: graph inventory as a dataframe and as serialised graph sets.
  - `export`, `export_gallery`: write graphs as HTML files, the latter with an `index.html`.
  - One method per graph type, named in `PLOT_METHOD_GRAPH_TYPES` (for example `composite_curve`, `grand_composite_curve`, `net_load_profiles`, `energy_transfer_diagram`), each taking `zone_name`, `index`, `show` and `return_graph_data`; the heat-pump, refrigeration and ORC variants are listed there too, and the heat-pump and refrigeration methods (`grand_composite_curve_with_heat_pump`, `..._with_refrigeration`, the `net_load_profiles_with_*` pair) also take `target` [problem.py:_PlotAccessor._plot_hpr].
- `PLOT_METHOD_GRAPH_TYPES`, `PlotlyJs` (`problem.py`): method name to `GraphType` map; the allowed `include_plotlyjs` values (`True`, `"cdn"`, `"directory"`).
- `graph_simple_cc_plot`, `plot_t_h_curve`, `plot_t_h_curve_with_piecewise_and_bounds` (`simple.py`): quick composite curve and T-H plots that call `show()`.
- Subdirectories: none.

## Invariants
- `zone_name` is matched against the graph key, name, zone name, zone address and target type, also by the part after the first `/` (selector and candidate alike); no match raises `KeyError` listing the zones. [problem.py:_graph_set_matches_zone_selector] [problem.py:_PlotAccessor._select_graphs]
- `graph_type` must be a `GraphType`; no match raises `ValueError`, an out-of-range `index` raises `IndexError`. [problem.py:_PlotAccessor._select_graph]
- `show=True` with `return_graph_data=True` raises `ValueError`. [problem.py:_PlotAccessor._plot_graph]
- The `plot` argument of the export methods must be a bound method of this problem's accessor. [problem.py:_PlotAccessor._graph_type_for_method]
- With no target selected, `data()` returns the all-periods graphs keyed `<key> [<period_id>]`; with none available it raises `RuntimeError`. [problem.py:_PlotAccessor.data]
- Heat-pump and refrigeration plots need a selector supplied by the application layer. [problem.py:_PlotAccessor._plot_hpr]
- Axes are heat flow in kW against temperature in degC; segment colours come from `LineColour`. [plotly.py:_apply_default_layout] [plotly.py:_SEGMENT_COLOURS]
- A legend entry shows once per series id. [plotly.py:_legend_details]
- Exported file stems are slugified from zone, graph type and position. [problem.py:_slugify]

## Side Effects
- Imports Plotly lazily; a missing install raises `ImportError` with an install message. [plotly.py:_require_plotly] [simple.py:_require_plotly]
- `export` and `export_gallery` create the output directory and write HTML (and a shared `plotly.min.js` for `"directory"`). [problem.py:_PlotAccessor.export_gallery]
- `show=True` and the `simple.py` functions open a Plotly viewer. [problem.py:_PlotAccessor._build_graph_figure] [simple.py:graph_simple_cc_plot]

## Verification
- Run: `uv run --no-sync pytest tests/presentation/test_plot_exports.py tests/presentation/test_simple_graphs.py`
