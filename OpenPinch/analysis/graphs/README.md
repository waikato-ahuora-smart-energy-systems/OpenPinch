# Graph Data

## Overview
Turns the problem tables and diagram payloads stored on targets into serialised graph sets (curves with coordinates, colours, arrows and series metadata). `service.py` walks a zone hierarchy, `specifications.py` declares which graphs exist and how each is built, and `composite.py` and `grand_composite.py` build the two curve families. The output feeds the presentation layer; nothing here draws.

## Interface
- `get_output_graph_data(zone, graph_sets)` (`service.py`): returns `{target name: graph set}` for every reportable target in the zone and its subzones.
- `resolve_graph_context`, `iter_available_graph_specs`, `build_available_graphs`, `build_graph_from_spec`, `build_graph_set_payload` (`service.py`): per-target steps of the build.
- `clean_composite_curve`, `clean_composite_curve_ends` (`composite.py`): remove constant-enthalpy ends, repeated points and collinear points from a curve.
- `GRAPH_BUILD_SPECS`, `COMPOSITE_GRAPH_SPECS`, `GCC_GRAPH_SPECS`, `ENERGY_TRANSFER_GRAPH_SPECS`, `REFRIGERATION_GRAPH_SPECS`, `ORC_GRAPH_SPECS` (`specifications.py`): declarative graph specifications.
- `GRAPH_SERIES_META` (`metadata.py`): labels, descriptions and titles keyed by problem table column.
- `GCC_VERTICAL_TOL` (`grand_composite.py`): enthalpy difference below which a GCC segment counts as vertical.
- `DECIMAL_PLACES` (`primitives.py`): rounding applied to serialised coordinates.
- Subdirectories: none.

## Invariants
- A graph is built only when its `GraphType` key exists in the target's `graphs`; output follows the order of `GRAPH_BUILD_SPECS`. [service.py:iter_available_graph_specs] [specifications.py:GRAPH_BUILD_SPECS]
- Only targets with `reportable` true are included. [service.py:get_output_graph_data]
- Serialised coordinates are rounded to 2 decimal places. [primitives.py:_create_curve]
- A series with no finite value above `tol`, or a composite series whose duty range is within the heat resolution, is not plotted. [primitives.py:_should_plot_series] [composite.py:_make_composite_graph]
- Composite curve ends are trimmed with a tolerance proportional to the curve's duty scale (`duty_tolerance`), not `np.isclose`'s relative tolerance on the end value. [composite.py:clean_composite_curve_ends]
- Arrow heads for the Total Site Profiles graph are the reverse of those for other composite graphs. [composite.py:_graph_cc]
- GCC segments are classified by the sign of the enthalpy step, inverted for utility profiles; near-vertical steps are drawn black. [grand_composite.py:_classify_segment] [grand_composite.py:_build_gcc_segments]
- `value_field` and its companion list (stream types or utility flags) must have equal length, otherwise `ValueError`. [primitives.py:_normalise_graph_values]
- Energy transfer diagram curves plot stacked heat on x against temperature on y. [service.py:_make_energy_transfer_diagram_graph]

## Side Effects
- Fills and returns the `graph_sets` dict passed in, and recurses into subzones with the same dict. [service.py:get_output_graph_data]
- No file IO or global state; builders return new dicts.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/test_graphs.py`
