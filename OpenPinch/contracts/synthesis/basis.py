"""The median design basis of a multiperiod HEN, for a preview before synthesis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _BasisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HenBasisCluster(_BasisModel):
    """Operating periods clustered together, with their weights and medoid."""

    period_ids: tuple[str, ...]
    weights: tuple[float, ...]
    weight_share: float
    medoid: str


class HenBasisStream(_BasisModel):
    """One stream of the basis case: the weighted medians, or why it is out.

    ``quantities`` holds ``t_supply``, ``t_target``, ``heat_flow``, ``htc``
    and ``dt_cont`` in ``units``. ``active_share`` is the share of the
    cluster's weight the stream runs in.
    """

    name: str
    zone: str
    included: bool
    active_share: float
    quantities: dict[str, float | None]
    units: dict[str, str]


class HenDesignBasis(_BasisModel):
    """Clusters of operating periods and the synthetic case HEN design uses.

    ``basis_cluster`` indexes ``clusters``: the heaviest cluster, whose
    synthetic case ``streams`` describes.
    """

    design_basis: Literal["median"]
    basis_clusters: int
    clusters: tuple[HenBasisCluster, ...]
    basis_cluster: int
    streams: tuple[HenBasisStream, ...]


__all__ = ["HenBasisCluster", "HenBasisStream", "HenDesignBasis"]
