"""Hand-built pinch sides for the heuristic pinch design tests."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.heuristic.side_streams import (
    PinchSide,
    SideStream,
)


def stream(kind, index, cp, load, t_pinch, *, at_pinch=True):
    """A single-period constant-CP stream ``load`` kW from its pinch end.

    Above the pinch both streams get hotter away from it, below it colder.
    """

    return kind, index, cp, load, t_pinch, at_pinch


def side(location, hot, cold, *, approach=10.0, heaters=1, coolers=1):
    """A one-period side with the given ``stream(...)`` tuples."""

    sign = 1.0 if location == "above" else -1.0

    def build(spec):
        kind, index, cp, load, t_pinch, at_pinch = spec
        return SideStream(
            kind=kind,
            index=index,
            loads=(float(load),),
            pinch_cps=(float(cp),),
            at_pinch=(at_pinch,),
            heats=((0.0, float(load)),),
            temperatures=((float(t_pinch), float(t_pinch) + sign * load / cp),),
            weights=(1.0,),
        )

    hot_streams = tuple(build(spec) for spec in hot)
    cold_streams = tuple(build(spec) for spec in cold)
    n_hot = max((s.index for s in hot_streams), default=-1) + 1
    n_cold = max((s.index for s in cold_streams), default=-1) + 1
    return PinchSide(
        location=location,
        required=True,
        hot=hot_streams,
        cold=cold_streams,
        approach=(
            tuple(tuple(approach for _j in range(n_cold)) for _i in range(n_hot)),
        ),
        match_allowed=tuple(tuple(True for _j in range(n_cold)) for _i in range(n_hot)),
        heater_allowed=tuple(
            tuple(int(location == "above") for _u in range(heaters))
            for _j in range(n_cold)
        ),
        cooler_allowed=tuple(
            tuple(int(location == "below") for _u in range(coolers))
            for _i in range(n_hot)
        ),
        hot_count=n_hot,
        cold_count=n_cold,
        weights=(1.0,),
    )
