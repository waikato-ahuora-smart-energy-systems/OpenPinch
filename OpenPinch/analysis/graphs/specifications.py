"""Private declarative graph build specifications."""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.enums import GraphType, ProblemTableLabel, StreamLoc


@dataclass(frozen=True)
class _GraphBuildSpec:
    """Declarative instructions for building one target graph payload."""

    graph_type: GraphType
    label: str
    builder: str
    value_fields: tuple = ()
    stream_types: tuple = ()
    utility_profile_flags: tuple = ()
    include_arrows: bool = True


COMPOSITE_GRAPH_SPECS = (
    _GraphBuildSpec(
        graph_type=GraphType.CC,
        label="Composite Curve",
        builder="composite",
        value_fields=(ProblemTableLabel.H_HOT, ProblemTableLabel.H_COLD),
        stream_types=(StreamLoc.HotS, StreamLoc.ColdS),
    ),
    _GraphBuildSpec(
        graph_type=GraphType.SCC,
        label="Shifted Composite Curve",
        builder="composite",
        value_fields=(ProblemTableLabel.H_HOT, ProblemTableLabel.H_COLD),
        stream_types=(StreamLoc.HotS, StreamLoc.ColdS),
    ),
    _GraphBuildSpec(
        graph_type=GraphType.BCC,
        label="Balanced Composite Curve",
        builder="composite",
        value_fields=(ProblemTableLabel.H_HOT_BAL, ProblemTableLabel.H_COLD_BAL),
        stream_types=(StreamLoc.HotS, StreamLoc.ColdS),
    ),
    _GraphBuildSpec(
        graph_type=GraphType.NLP,
        label="Net Load Curves",
        builder="composite",
        value_fields=(
            ProblemTableLabel.H_NET_HOT,
            ProblemTableLabel.H_NET_COLD,
            ProblemTableLabel.H_HOT_UT,
            ProblemTableLabel.H_COLD_UT,
            ProblemTableLabel.H_HOT_HP,
            ProblemTableLabel.H_COLD_HP,
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
    _GraphBuildSpec(
        graph_type=GraphType.NLP_HP,
        label="Net Load Profiles with Heat Pump",
        builder="composite",
        value_fields=(
            ProblemTableLabel.H_NET_HOT,
            ProblemTableLabel.H_NET_COLD,
            ProblemTableLabel.H_HOT_HP,
            ProblemTableLabel.H_COLD_HP,
        ),
        stream_types=(
            StreamLoc.HotS,
            StreamLoc.ColdS,
            StreamLoc.HotU,
            StreamLoc.ColdU,
        ),
    ),
    _GraphBuildSpec(
        graph_type=GraphType.NLP_X,
        label="Exergetic Net Load Profiles",
        builder="composite",
        value_fields=(ProblemTableLabel.X_SUR, ProblemTableLabel.X_DEF),
        stream_types=(StreamLoc.HotS, StreamLoc.ColdS),
    ),
    _GraphBuildSpec(
        graph_type=GraphType.TSP,
        label="Total Site Profiles",
        builder="composite",
        value_fields=(
            ProblemTableLabel.H_HOT,
            ProblemTableLabel.H_COLD,
            ProblemTableLabel.H_HOT_UT,
            ProblemTableLabel.H_COLD_UT,
        ),
        stream_types=(
            StreamLoc.HotS,
            StreamLoc.ColdS,
            StreamLoc.HotU,
            StreamLoc.ColdU,
        ),
    ),
)

GCC_GRAPH_SPECS = (
    _GraphBuildSpec(
        graph_type=GraphType.GCC,
        label="Grand Composite Curve",
        builder="gcc",
        value_fields=(
            ProblemTableLabel.H_NET,
            ProblemTableLabel.H_NET_NP,
            ProblemTableLabel.H_NET_V,
            ProblemTableLabel.H_NET_A,
            ProblemTableLabel.H_NET_UT,
        ),
        utility_profile_flags=(False, False, False, False, True),
    ),
    _GraphBuildSpec(
        graph_type=GraphType.GCC_R,
        label="Grand Composite Curve (Real)",
        builder="gcc",
        value_fields=(ProblemTableLabel.H_NET, ProblemTableLabel.H_NET_UT),
        utility_profile_flags=(False, True),
    ),
    _GraphBuildSpec(
        graph_type=GraphType.GCC_X,
        label="Exergetic Grand Composite Curve",
        builder="gcc",
        value_fields=(ProblemTableLabel.X_GCC,),
        utility_profile_flags=(False,),
    ),
    _GraphBuildSpec(
        graph_type=GraphType.SUGCC,
        label="Site Utility Grand Composite Curve",
        builder="gcc",
        value_fields=(ProblemTableLabel.H_NET_UT,),
        utility_profile_flags=(True,),
    ),
    _GraphBuildSpec(
        graph_type=GraphType.GCC_HP,
        label="Grand Composite Curve with Heat Pump",
        builder="gcc",
        value_fields=(ProblemTableLabel.H_NET_W_AIR, ProblemTableLabel.H_NET_HP),
        utility_profile_flags=(False, True),
    ),
)

ENERGY_TRANSFER_GRAPH_SPECS = (
    _GraphBuildSpec(
        graph_type=GraphType.ETD,
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
