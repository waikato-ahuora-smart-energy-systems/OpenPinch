"""Private declarative graph build specifications."""

from __future__ import annotations

from dataclasses import dataclass

from ....lib.enums import GT, PT, StreamLoc


@dataclass(frozen=True)
class GraphBuildSpec:
    """Declarative instructions for building one target graph payload."""

    graph_type: GT
    label: str
    builder: str
    value_fields: tuple = ()
    stream_types: tuple = ()
    utility_profile_flags: tuple = ()
    include_arrows: bool = True


COMPOSITE_GRAPH_SPECS = (
    GraphBuildSpec(
        graph_type=GT.CC,
        label="Composite Curve",
        builder="composite",
        value_fields=(PT.H_HOT, PT.H_COLD),
        stream_types=(StreamLoc.HotS, StreamLoc.ColdS),
    ),
    GraphBuildSpec(
        graph_type=GT.SCC,
        label="Shifted Composite Curve",
        builder="composite",
        value_fields=(PT.H_HOT, PT.H_COLD),
        stream_types=(StreamLoc.HotS, StreamLoc.ColdS),
    ),
    GraphBuildSpec(
        graph_type=GT.BCC,
        label="Balanced Composite Curve",
        builder="composite",
        value_fields=(PT.H_HOT_BAL, PT.H_COLD_BAL),
        stream_types=(StreamLoc.HotS, StreamLoc.ColdS),
    ),
    GraphBuildSpec(
        graph_type=GT.NLP,
        label="Net Load Curves",
        builder="composite",
        value_fields=(
            PT.H_NET_HOT,
            PT.H_NET_COLD,
            PT.H_HOT_UT,
            PT.H_COLD_UT,
            PT.H_HOT_HP,
            PT.H_COLD_HP,
        ),
        stream_types=(
            StreamLoc.HotS,
            StreamLoc.ColdS,
            StreamLoc.HotU,
            StreamLoc.ColdU,
            StreamLoc.HotU,
            StreamLoc.ColdU,
        ),
    ),
    GraphBuildSpec(
        graph_type=GT.NLP_HP,
        label="Net Load Profiles with Heat Pump",
        builder="composite",
        value_fields=(PT.H_NET_HOT, PT.H_NET_COLD, PT.H_HOT_HP, PT.H_COLD_HP),
        stream_types=(
            StreamLoc.HotS,
            StreamLoc.ColdS,
            StreamLoc.HotU,
            StreamLoc.ColdU,
        ),
    ),
    GraphBuildSpec(
        graph_type=GT.NLP_X,
        label="Exergetic Net Load Profiles",
        builder="composite",
        value_fields=(PT.X_SUR, PT.X_DEF),
        stream_types=(StreamLoc.HotS, StreamLoc.ColdS),
    ),
    GraphBuildSpec(
        graph_type=GT.TSP,
        label="Total Site Profiles",
        builder="composite",
        value_fields=(PT.H_HOT, PT.H_COLD, PT.H_HOT_UT, PT.H_COLD_UT),
        stream_types=(
            StreamLoc.HotS,
            StreamLoc.ColdS,
            StreamLoc.HotU,
            StreamLoc.ColdU,
        ),
    ),
)

GCC_GRAPH_SPECS = (
    GraphBuildSpec(
        graph_type=GT.GCC,
        label="Grand Composite Curve",
        builder="gcc",
        value_fields=(PT.H_NET, PT.H_NET_NP, PT.H_NET_V, PT.H_NET_A, PT.H_NET_UT),
        utility_profile_flags=(False, False, False, False, True),
    ),
    GraphBuildSpec(
        graph_type=GT.GCC_R,
        label="Grand Composite Curve (Real)",
        builder="gcc",
        value_fields=(PT.H_NET, PT.H_NET_UT),
        utility_profile_flags=(False, True),
    ),
    GraphBuildSpec(
        graph_type=GT.GCC_X,
        label="Exergetic Grand Composite Curve",
        builder="gcc",
        value_fields=(PT.X_GCC,),
        utility_profile_flags=(False,),
    ),
    GraphBuildSpec(
        graph_type=GT.SUGCC,
        label="Site Utility Grand Composite Curve",
        builder="gcc",
        value_fields=(PT.H_NET_UT,),
        utility_profile_flags=(True,),
    ),
    GraphBuildSpec(
        graph_type=GT.GCC_HP,
        label="Grand Composite Curve with Heat Pump",
        builder="gcc",
        value_fields=(PT.H_NET_W_AIR, PT.H_NET_HP),
        utility_profile_flags=(False, True),
    ),
)

ENERGY_TRANSFER_GRAPH_SPECS = (
    GraphBuildSpec(
        graph_type=GT.ETD,
        label="Energy Transfer Diagram",
        builder="energy_transfer",
    ),
)

GRAPH_BUILD_SPECS = (
    COMPOSITE_GRAPH_SPECS[0],
    COMPOSITE_GRAPH_SPECS[1],
    COMPOSITE_GRAPH_SPECS[2],
    GCC_GRAPH_SPECS[0],
    GCC_GRAPH_SPECS[1],
    GCC_GRAPH_SPECS[2],
    COMPOSITE_GRAPH_SPECS[3],
    COMPOSITE_GRAPH_SPECS[4],
    COMPOSITE_GRAPH_SPECS[5],
    COMPOSITE_GRAPH_SPECS[6],
    GCC_GRAPH_SPECS[3],
    GCC_GRAPH_SPECS[4],
    ENERGY_TRANSFER_GRAPH_SPECS[0],
)
