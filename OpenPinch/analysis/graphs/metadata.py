"""Metadata registry for keyed graph series labels and titles."""

from dataclasses import dataclass
from typing import Optional

from ...domain.enums import PT, LegendSeries, StreamLoc


@dataclass(frozen=True)
class GraphSeriesMeta:
    """Display metadata for keyed graph series rendered in output graphs."""

    label: str
    description: str
    preferred_stream_loc: Optional[StreamLoc] = None
    composite_title: Optional[str] = None


GRAPH_SERIES_META: dict[str, GraphSeriesMeta] = {
    PT.H_HOT.value: GraphSeriesMeta(
        label="Hot CC",
        description="Hot Composite Curve",
        composite_title="Hot CC",
    ),
    PT.H_COLD.value: GraphSeriesMeta(
        label="Cold CC",
        description="Cold Composite Curve",
        composite_title="Cold CC",
    ),
    PT.H_HOT_BAL.value: GraphSeriesMeta(
        label="Hot Balanced CC",
        description="Hot Balanced Composite Curve",
        composite_title="Hot Balanced CC",
    ),
    PT.H_COLD_BAL.value: GraphSeriesMeta(
        label="Cold Balanced CC",
        description="Cold Balanced Composite Curve",
        composite_title="Cold Balanced CC",
    ),
    PT.H_NET_HOT.value: GraphSeriesMeta(
        label="Hot Net Load",
        description="Hot Net Load",
        composite_title="Hot Net Load",
    ),
    PT.H_NET_COLD.value: GraphSeriesMeta(
        label="Cold Net Load",
        description="Cold Net Load",
        composite_title="Cold Net Load",
    ),
    PT.H_HOT_UT.value: GraphSeriesMeta(
        label="Hot Utility",
        description="Hot Utility",
        composite_title="Hot Utility",
    ),
    PT.H_COLD_UT.value: GraphSeriesMeta(
        label="Cold Utility",
        description="Cold Utility",
        composite_title="Cold Utility",
    ),
    PT.H_NET.value: GraphSeriesMeta(
        LegendSeries.GCC.name,
        LegendSeries.GCC.value,
    ),
    PT.H_NET_NP.value: GraphSeriesMeta(
        LegendSeries.GCC_N.name,
        LegendSeries.GCC_N.value,
    ),
    PT.H_NET_V.value: GraphSeriesMeta(
        LegendSeries.GCC_V.name,
        LegendSeries.GCC_V.value,
    ),
    PT.H_NET_A.value: GraphSeriesMeta(
        LegendSeries.GCC_A.name,
        LegendSeries.GCC_A.value,
    ),
    PT.H_NET_UT.value: GraphSeriesMeta(
        LegendSeries.GCC_U.name,
        LegendSeries.GCC_U.value,
        StreamLoc.HotU,
    ),
    PT.H_HOT_HP.value: GraphSeriesMeta(
        label="Heat Pump Condenser",
        description="Heat Pump Condenser",
        composite_title="Heat Pump Condenser",
    ),
    PT.H_COLD_HP.value: GraphSeriesMeta(
        label="Heat Pump Evaporator",
        description="Heat Pump Evaporator",
        composite_title="Heat Pump Evaporator",
    ),
    PT.X_GCC.value: GraphSeriesMeta(
        label="GCC_X",
        description="Exergetic Grand Composite Curve",
    ),
    PT.X_SUR.value: GraphSeriesMeta(
        label="Exergy Surplus",
        description="Exergy Surplus",
        composite_title="Exergy Surplus",
    ),
    PT.X_DEF.value: GraphSeriesMeta(
        label="Exergy Deficit",
        description="Exergy Deficit",
        composite_title="Exergy Deficit",
    ),
}
