"""A synthetic single-period case from clustered operating periods.

Works on problem data (``PinchProblem.to_problem_json()``), so the case it
builds is an ordinary single-period problem that targeting and synthesis run
on unchanged. Operating periods (weight above zero, some process stream
running) are clustered on each stream's supply and target temperature, heat
flow and running flag, each scaled by its period-weighted spread. A cluster's
case takes, per stream, the period-weighted median of its supply and target
temperatures, heat flow, film coefficient and ``dt_cont`` over the member
periods it runs in, and the medoid period's utilities. A stream running in
less than ``active_share`` of the cluster's weight is left out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clustering import weighted_k_medoids

# Quantities each stream's synthetic value is the weighted median of.
MEDIAN_QUANTITIES = ("t_supply", "t_target", "heat_flow", "htc", "dt_cont")
# Weight of a stream's running flag against its scaled temperatures and flow.
_RUNNING_WEIGHT = 1.0
SYNTHETIC_PERIOD_ID = "median"


def period_source(value, index: int, period_id: str):
    """``value`` with every per-period quantity reduced to one period's value.

    A ``values`` list is read by its ``period_ids`` when it carries them, else
    by period position; a one-value list applies to every period.
    """

    if isinstance(value, list):
        return [period_source(item, index, period_id) for item in value]
    if not isinstance(value, dict):
        return value
    if "values" in value:
        values = value["values"]
        ids = value.get("period_ids")
        position = (
            [str(item) for item in ids].index(period_id)
            if ids is not None
            else (index if len(values) > 1 else 0)
        )
        reduced = {
            key: item
            for key, item in value.items()
            if key not in ("values", "period_ids", "weights")
        }
        return reduced | {"value": period_source(values[position], index, period_id)}
    return {key: period_source(item, index, period_id) for key, item in value.items()}


@dataclass(frozen=True)
class BasisCluster:
    """Member periods (by position), their weights and the medoid's position."""

    members: tuple[int, ...]
    weights: tuple[float, ...]
    medoid: int

    @property
    def weight(self) -> float:
        return float(sum(self.weights))


@dataclass(frozen=True)
class BasisStream:
    """One stream of a cluster's case: its values and whether it is in."""

    name: str
    zone: str
    included: bool
    active_share: float
    quantities: dict[str, float | None]
    units: dict[str, str]


@dataclass(frozen=True)
class MedianBasis:
    """Clusters of operating periods and each cluster's synthetic case."""

    period_ids: tuple[str, ...]
    clusters: tuple[BasisCluster, ...]
    basis_cluster: int
    streams: tuple[tuple[BasisStream, ...], ...]
    sources: tuple[dict, ...]


def median_basis(
    source: dict,
    period_ids,
    weights,
    *,
    clusters: int,
    active_share: float,
) -> MedianBasis:
    """Cluster ``source``'s operating periods and build each cluster's case."""

    period_ids = tuple(str(item) for item in period_ids)
    weights = np.asarray(weights, dtype=float)
    for record in source["streams"]:
        if record.get("segments") or record.get("profile"):
            raise ValueError(
                f"the median design basis does not support stream "
                f"{record['name']!r}, which has segments or a profile; use "
                "design_basis='period' or 'all'."
            )
    periods = [
        period_source(source, index, period_id)
        for index, period_id in enumerate(period_ids)
    ]
    values = {
        quantity: np.array(
            [
                [_number(record.get(quantity)) for record in period["streams"]]
                for period in periods
            ],
            dtype=float,
        )
        for quantity in MEDIAN_QUANTITIES
    }
    running = np.array(
        [
            [
                bool(record.get("active", True))
                and _number(record.get("heat_flow")) > 0
                for record in period["streams"]
            ]
            for period in periods
        ]
    )
    operating = [
        index
        for index in range(len(period_ids))
        if weights[index] > 0 and running[index].any()
    ]
    if not operating:
        raise ValueError("the median design basis needs an operating period.")
    if not 1 <= clusters <= len(operating):
        raise ValueError(
            f"basis_clusters must be between 1 and {len(operating)}, the number "
            f"of operating periods; got {clusters}."
        )
    features = _scaled_features(values, running, weights, operating)
    clustering = weighted_k_medoids(features, weights[operating], clusters)
    found = tuple(
        BasisCluster(
            members=members,
            weights=tuple(float(weights[index]) for index in members),
            medoid=operating[medoid],
        )
        for position, medoid in enumerate(clustering.medoids)
        for members in [
            tuple(
                operating[row]
                for row, label in enumerate(clustering.labels)
                if label == position
            )
        ]
    )
    # The heaviest cluster, ties going to the one whose medoid comes first.
    basis_cluster = min(
        range(len(found)), key=lambda n: (-found[n].weight, found[n].medoid)
    )
    cases = [
        _cluster_case(source, periods, values, running, weights, cluster, active_share)
        for cluster in found
    ]
    return MedianBasis(
        period_ids=period_ids,
        clusters=found,
        basis_cluster=basis_cluster,
        streams=tuple(streams for streams, _case in cases),
        sources=tuple(case for _streams, case in cases),
    )


def _number(value) -> float:
    if isinstance(value, dict):
        value = value.get("value")
    return float("nan") if value is None else float(value)


def _scaled_features(values, running, weights, operating) -> np.ndarray:
    """Per stream: supply, target, heat flow and running flag, as columns.

    Each column is divided by its period-weighted standard deviation over the
    operating periods, so a rarely run period does not stretch the scale; a
    column with no spread is left as it is, and contributes nothing.
    """

    rows = np.asarray(operating)
    w = weights[rows] / weights[rows].sum()
    columns = [
        np.nan_to_num(values[quantity][rows])
        for quantity in ("t_supply", "t_target", "heat_flow")
    ] + [running[rows].astype(float)]
    scaled = []
    for column in columns:
        mean = w @ column
        spread = np.sqrt(w @ (column - mean) ** 2)
        scaled.append(column / np.where(spread > 0, spread, 1.0))
    scaled[-1] = scaled[-1] * _RUNNING_WEIGHT
    return np.hstack(scaled)


def _weighted_median(values, weights) -> float:
    """The lower weighted median: the first value reaching half the weight."""

    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(
        np.asarray(values)[order][np.searchsorted(cumulative, cumulative[-1] / 2)]
    )


def _cluster_case(source, periods, values, running, weights, cluster, active_share):
    medoid = periods[cluster.medoid]
    members = np.asarray(cluster.members)
    member_weight = weights[members].sum()
    case = {key: value for key, value in medoid.items() if key != "streams"}
    case_streams, rows = [], []
    for stream, record in enumerate(medoid["streams"]):
        on = members[running[members, stream]]
        share = float(weights[on].sum() / member_weight)
        included = share >= active_share and len(on) > 0
        quantities: dict[str, float | None] = {}
        units = {}
        for quantity in MEDIAN_QUANTITIES:
            unit = (source["streams"][stream].get(quantity) or {}).get("unit")
            column = values[quantity][on, stream] if included else np.array([])
            if not included or np.isnan(column).any():
                quantities[quantity] = (
                    None
                    if np.isnan(_number(record.get(quantity)))
                    else _number(record.get(quantity))
                )
            else:
                quantities[quantity] = _weighted_median(column, weights[on])
            if unit is not None:
                units[quantity] = unit
        rows.append(
            BasisStream(
                name=record["name"],
                zone=record.get("zone", ""),
                included=included,
                active_share=share,
                quantities=quantities,
                units=units,
            )
        )
        if not included:
            continue
        synthetic = dict(record) | {"heat_capacity_flowrate": None}
        for quantity, value in quantities.items():
            if value is not None and quantity in units:
                synthetic[quantity] = {"value": value, "unit": units[quantity]}
        case_streams.append(synthetic)
    case["streams"] = case_streams
    case["options"] = dict(case["options"]) | {
        "PROBLEM_PERIOD_IDS": [SYNTHETIC_PERIOD_ID],
        "PROBLEM_PERIOD_WEIGHTS": [1.0],
    }
    return tuple(rows), case


__all__ = [
    "MEDIAN_QUANTITIES",
    "SYNTHETIC_PERIOD_ID",
    "BasisCluster",
    "BasisStream",
    "MedianBasis",
    "median_basis",
    "period_source",
]
