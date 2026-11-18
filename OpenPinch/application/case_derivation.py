"""Application-owned case derivation with independent runtime ownership."""

from copy import deepcopy
from dataclasses import dataclass

from ..domain.enums import ZoneType
from ._problem.input.utilities import _set_utilities_for_zone_and_subzones
from ._problem.targeting.state import snapshot_problem


@dataclass(frozen=True)
class SiteProcess:
    """One process of a Total Site case and the keys of its process streams."""

    name: str
    address: str
    hot_streams: tuple[str, ...]
    cold_streams: tuple[str, ...]

    @property
    def one_sided(self) -> bool:
        """Whether the process has only hot or only cold streams."""
        return not self.hot_streams or not self.cold_streams


def _site_keys(site_collection, process_collection) -> tuple[str, ...]:
    """The site's keys of the streams a process holds, in site order."""
    held = {id(stream) for _, stream in process_collection.items()}
    return tuple(key for key, stream in site_collection.items() if id(stream) in held)


def site_processes(problem, zone=None) -> tuple[SiteProcess, ...]:
    """The processes of a Site zone (the root by default), in zone order.

    Every subzone of the site must be a Process Zone; a process holds the
    streams of its own subzones, and a stream placed on the site zone itself
    is prepared into a Process Zone of its own. Stream keys are the site
    zone's, as its solver arrays use them.
    """
    site = problem._resolve_target_zone(zone)
    if site.type != ZoneType.S.value:
        raise ValueError(f"{site.address!r} is a {site.type}, not a Site zone.")
    processes = []
    for subzone in site.subzones.values():
        if subzone.type != ZoneType.P.value:
            raise ValueError(
                f"{subzone.address!r} is a {subzone.type}; every subzone of a "
                "site must be a Process Zone."
            )
        processes.append(
            SiteProcess(
                name=subzone.name,
                address=subzone.address,
                hot_streams=_site_keys(site.hot_streams, subzone.hot_streams),
                cold_streams=_site_keys(site.cold_streams, subzone.cold_streams),
            )
        )
    return tuple(processes)


def _within(address: str, ancestor: str) -> bool:
    return address == ancestor or address.startswith(ancestor + "/")


def _tree_along(tree: dict, path: list[str]) -> dict:
    """The zone tree reduced to the zones on ``path`` and the last one's subtree."""
    if len(path) == 1:
        return tree
    children = [
        _tree_along(child, path[1:])
        for child in tree.get("children") or ()
        if child["name"] == path[1]
    ]
    return tree | {"children": children}


def process_case(problem, zone, *, project_name=None):
    """A new problem holding one Process Zone's streams with every utility.

    The zones above the process are kept, without their other subzones, so
    stream keys match the source problem's. Plant profile data and any
    network are not carried over.
    """
    process = problem._resolve_target_zone(zone)
    if process.type != ZoneType.P.value:
        raise ValueError(
            f"{process.address!r} is a {process.type}, not a Process Zone."
        )
    if problem._process_components:
        raise ValueError(
            "a process case cannot be derived from a problem with process components."
        )
    source = problem.to_problem_json()
    source["streams"] = [
        record
        for record in source["streams"]
        if _within(record["zone"], process.address)
    ]
    source["zone_tree"] = _tree_along(source["zone_tree"], process.address.split("/"))
    source["network"] = None
    source["plant_profile_data"] = []
    return type(problem)(
        source,
        project_name=problem.project_name if project_name is None else project_name,
    )


def _align_period_values(value, donor_ids, receiver_ids):
    if isinstance(value, list):
        return [_align_period_values(item, donor_ids, receiver_ids) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: _align_period_values(item, donor_ids, receiver_ids)
        for key, item in value.items()
    }
    if "values" in value and "period_ids" not in value:
        values = value["values"]
        if len(values) == len(donor_ids):
            order = [donor_ids.index(sid) for sid in receiver_ids]
            result["values"] = [values[i] for i in order]
            if value.get("weights") is not None:
                result["weights"] = [value["weights"][i] for i in order]
    return result


def _bind_implicit_units(value, donor_config, receiver_config):
    """Keep unlabelled quantities physical when case unit defaults differ."""
    from ..contracts.units import (
        INPUT_UNIT_RULES,
        standardise_input_value,
        without_period_ids,
    )

    if isinstance(value, list):
        return [
            _bind_implicit_units(item, donor_config, receiver_config) for item in value
        ]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        field = {"temperature": "t_supply", "cumulative_heat": "heat_flow"}.get(
            key, key
        )
        if (
            field in INPUT_UNIT_RULES
            and item is not None
            and not (isinstance(item, dict) and item.get("unit"))
        ):
            # The ids are carried over below; only magnitudes are converted.
            magnitudes = without_period_ids(item)
            source = standardise_input_value(
                magnitudes, field_name=field, config=donor_config
            )
            local = standardise_input_value(
                magnitudes, field_name=field, config=receiver_config
            )
            if source.to_dict() != local.to_dict():
                converted = source.to_dict()
                if isinstance(item, dict):
                    if "values" in item:
                        # Even a singleton vector must retain its array shape
                        # when it carries period identities or weights.
                        converted.pop("value", None)
                        converted["values"] = source.period_values.tolist()
                    for attribute in ("weights", "period_ids"):
                        if attribute in item:
                            converted[attribute] = item[attribute]
                result[key] = converted
                continue
        result[key] = _bind_implicit_units(item, donor_config, receiver_config)
    return result


def with_utilities_from(receiver, donor, *, project_name=None):
    from .problem import PinchProblem

    if not isinstance(donor, PinchProblem):
        raise TypeError("other_problem must be a prepared PinchProblem.")
    receiver_ids, donor_ids = list(receiver.period_ids), list(donor.period_ids)
    if set(receiver_ids) != set(donor_ids):
        raise ValueError("Utility transfer requires matching period sets.")
    source = receiver.to_problem_json()
    source["utilities"] = _align_period_values(
        _bind_implicit_units(
            donor.to_problem_json()["utilities"],
            donor._require_prepared_root_zone().config,
            receiver._require_prepared_root_zone().config,
        ),
        donor_ids,
        receiver_ids,
    )
    prepared = type(receiver)(
        source,
        project_name=receiver.project_name if project_name is None else project_name,
    )
    if not receiver._process_components:
        return prepared
    # The connected graph preserves component stream memberships and redirects
    # component.problem in one deepcopy. Canonical preparation validates utilities.
    derived = snapshot_problem(receiver)
    derived._problem_data = deepcopy(prepared._problem_data)
    derived._validated_data = deepcopy(prepared._validated_data)
    derived._problem_filepath = None
    derived._input_source_kind = prepared._input_source_kind
    derived._validation_context = deepcopy(prepared._validation_context)
    derived._analysis_owner_id = prepared._analysis_owner_id
    derived.project_name = prepared.project_name
    _set_utilities_for_zone_and_subzones(
        derived._master_zone,
        prepared._master_zone.hot_utilities,
        prepared._master_zone.cold_utilities,
    )
    derived._invalidate_analysis()
    return derived


def single_period_case(problem, period_id, *, project_name=None):
    """A new single-period problem holding ``period_id``'s data alone.

    Every per-period quantity takes that period's value; the period keeps its
    id and a weight of one. Synthesis and targeting run on it unchanged.
    """
    period_ids = list(problem.period_ids)
    period_id = str(period_id)
    if period_id not in period_ids:
        raise ValueError(
            f"unknown period {period_id!r}; the problem has periods "
            + ", ".join(repr(item) for item in period_ids)
            + "."
        )
    if problem._process_components:
        raise ValueError(
            "a single-period case cannot be derived from a problem with "
            "process components."
        )
    from ..analysis.heat_exchanger_networks.basis.synthetic_case import period_source

    source = period_source(
        problem.to_problem_json(), period_ids.index(period_id), period_id
    )
    source["options"] |= {
        "PROBLEM_PERIOD_IDS": [period_id],
        "PROBLEM_PERIOD_WEIGHTS": [1.0],
    }
    return type(problem)(
        source,
        project_name=problem.project_name if project_name is None else project_name,
    )
