# HPR Shared Helpers (private)

## Overview
Private helper modules behind `common/shared.py` and the targeting and cycle
modules of the heat-pump package. They hold three independent pieces: the
placement of ambient air and cooling water in a heat cascade (`air.py`), the
conversion of solved Carnot or ambient duties into utility streams
(`streams.py`), and a Plotly profile plot for debugging (`plotting.py`). The
`__init__.py` has only a docstring.

## Interface
What the parent and siblings import from this package:
- `cascade_with_air`, `CascadeWithAir` (`air.py`): place free air, then optional cooling water, in a cascade and return external top and bottom duties and the air and water duties; `CascadeWithAir` is a frozen dataclass.
- `ambient_source_temperature`, `ambient_sink_temperature`, `cooling_water_sink_temperature` (`air.py`): shifted temperatures at which air supplies or takes heat and at which cooling water takes heat (`None` when no cooling water is defined).
- `get_Q_vals_at_T_hpr_from_bckgrd_profile` (`streams.py`): stage duties read from a background T-H profile at proposed HPR temperatures.
- `get_carnot_hpr_cycle_streams` (`streams.py`): one `StreamCollection` of latent condenser (hot) and evaporator (cold) streams for a Carnot summary.
- `get_ambient_air_stream` (`streams.py`): `AIR_H*` and `AIR_C*` process streams for the ambient heat given and taken by a solved result.
- `plot_multi_hp_profiles_from_results` (`plotting.py`): Plotly figure of background sink and source curves plus condenser and evaporator streams.
- Subdirectories: none.

## Invariants
- Air and cooling water never create a utility demand: the duties they take or give are limited by the smallest cascade flow on the affected side, and cannot be negative. [air.py:cascade_with_air]
- The cascade is supplied with `T` descending and `H_net` clipped to non-negative before use. [air.py:cascade_with_air]
- Air supplies heat before it takes heat, and cooling water acts only on what air leaves. [air.py:cascade_with_air]
- A sink at or above the top of the cascade takes no heat. [air.py:_sink_duty]
- Ambient and cooling-water levels are shifted by `dt_env_cont`, `dt_cooling_water` and `dtcont_hp`, with `dtcont_hp` applied in the opposite sense on the source and sink sides. [air.py:ambient_source_temperature]
- Carnot condenser and evaporator levels closer than `dt_phase_change` are merged and their duties summed before streams are built. [streams.py:_get_carnot_hpr_cycle_cascade_profile]
- Hot streams run from `T` down to `T - dt_phase_change`; cold streams run from `T - dt_phase_change` up to `T`. [streams.py:_build_latent_streams]
- Ambient air streams are only created for duties above `tol`, and are marked as process streams with the `dt_env_cont` contribution; Carnot cycle streams are not. [streams.py:get_ambient_air_stream]
- Stage duties from a background profile are clipped at zero. [streams.py:get_Q_vals_at_T_hpr_from_bckgrd_profile]

## Side Effects
- Optional import of Plotly at call time; a missing package raises `ImportError` with extras guidance. [plotting.py:_require_plotly]
- Builds problem tables through `analysis.targeting.cascade` for the plot. [plotting.py:_get_hpr_cascade]
- Everything else is pure computation on arguments. No file or network IO, no global state. [air.py:cascade_with_air]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_pumps/test_air_cascade.py` (air placement). Streams and plotting are covered by `tests/analysis/heat_pumps/test_shared.py`.
