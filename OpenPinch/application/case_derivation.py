"""Application-owned case derivation with independent runtime ownership."""

from copy import deepcopy

from ..analysis.heat_pumps.common._shared.air import UTILITY_BAND_MAX_SPAN_K
from ..domain.enums import StreamType, ZoneType
from ..domain.targets import DirectHeatPumpTarget, IndirectHeatPumpTarget
from ._problem.input.utilities import _set_utilities_for_zone_and_subzones
from ._problem.targeting.state import snapshot_problem


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


def _ancestry(zone) -> list[str]:
    """Zone names from the root down to ``zone``; a name may contain "/"."""
    names = []
    while zone is not None:
        names.append(zone.name)
        zone = zone.parent_zone
    return names[::-1]


def _zone_case(problem, zone):
    """A new problem holding one zone's streams with every utility.

    The zones above it are kept, without their other subzones, so stream keys
    match the source problem's. The utilities are the source problem's
    prepared ones, defaults included: preparation places a default from the
    streams' temperatures, which the zone's streams alone would change. Plant
    profile data and any network are not carried over.
    """
    if problem._process_components:
        raise ValueError(
            "a zone case cannot be derived from a problem with process components."
        )
    source = problem.to_problem_json()
    source["streams"] = [
        record for record in source["streams"] if _within(record["zone"], zone.address)
    ]
    source["zone_tree"] = _tree_along(source["zone_tree"], _ancestry(zone))
    source["network"] = None
    source["plant_profile_data"] = []
    case = type(problem)(source, project_name=problem.project_name)
    site = problem._require_prepared_root_zone()
    _set_utilities_for_zone_and_subzones(
        case._require_prepared_root_zone(),
        deepcopy(site.hot_utilities),
        deepcopy(site.cold_utilities),
    )
    case._invalidate_analysis()
    return case


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


def _value_record(value) -> dict:
    """A value as an input record: one number, or one per period."""
    record = value.to_dict()
    if "values" in record:
        return {"values": [float(v) for v in record["values"]], "unit": record["unit"]}
    return {"value": float(record["value"]), "unit": record["unit"]}


def _heat_pump_stream_record(stream, *, zone: str, dt_cont: dict, mark: dict) -> dict:
    """One heat pump stream of a target as a process stream input record.

    A segmented stream (a vapour compression condenser: desuperheating,
    condensing, subcooling) keeps its segments, each with its own film
    coefficient; every segment takes the stream's contribution.
    """
    record = {"zone": zone, "name": stream.name, "dt_cont": dt_cont, "heat_pump": mark}
    if not stream.has_segments:
        return record | _thermal_record(stream)
    return record | {
        "segments": [
            {"name": segment.name} | _thermal_record(segment)
            for segment in stream.segments
        ]
    }


def _thermal_record(part) -> dict:
    """The temperatures, duty and film coefficient of a stream or segment."""
    return {
        "t_supply": _value_record(part.supply_temperature),
        "t_target": _value_record(part.target_temperature),
        "heat_flow": _value_record(part.heat_flow),
        "htc": _value_record(part.heat_transfer_coefficient),
    }


def _heat_pump_zone_address(problem, target, zone) -> str:
    """The process zone a heat pump target's streams are added to: the
    target's own zone, or the process zone under it that ``zone`` names
    (by name or address), or the only one."""
    root = problem._require_prepared_root_zone()
    zones = {z.address: z for z in problem._walk_zone_tree(root)}
    scope = target.provenance.zone_address
    if zones[scope].type == ZoneType.P.value:
        return scope
    below = [
        address
        for address, z in zones.items()
        if _within(address, scope) and z.type == ZoneType.P.value
    ]
    if zone is not None:
        named = [address for address in below if zone in (address, zones[address].name)]
        if not named:
            raise ValueError(
                f"The heat pump target of {scope!r} covers no process zone "
                f"{zone!r}. It covers: "
                + ", ".join(repr(address) for address in below)
                + "."
            )
        below = named
    if len(below) != 1:
        raise ValueError(
            f"The heat pump target of {scope!r} covers {len(below)} process "
            "zones; say which takes its streams with zone=. They are: "
            + ", ".join(repr(address) for address in below)
            + "."
        )
    return below[0]


def with_heat_pump_streams(
    problem, target, *, zone=None, machine="HP", project_name=None
):
    """A new problem holding ``problem``'s streams and a heat pump target's.

    Each condenser stream of ``target`` is added as a hot stream and each
    evaporator stream as a cold stream, marked with ``machine`` and its
    role. They carry ``HPR_DT_CONT``, the approach the target assumed
    between the heat pump and the process, so the new case's direct
    integration targets are the heat pump target's utilities.
    """
    from .hpr_selection import validate_hpr_target

    _require_heat_pump_stream_support(problem)
    validate_hpr_target(problem, target)
    if isinstance(target, IndirectHeatPumpTarget):
        if len(problem.period_ids) > 1:
            raise ValueError(
                "The zones of a heat pump on the utility system are not built "
                f"for a problem with {len(problem.period_ids)} periods yet; "
                "derive a single-period case and target the heat pump on it."
            )
        case = _case_with_utility_heat_pump_zones(
            problem,
            target,
            dt_cont=float(target.provenance.effective_settings["HPR_DT_CONT"]),
            machine=machine,
            project_name=project_name,
        )
        _require_heat_pump_site_target(case, target)
        return case
    if not isinstance(target, DirectHeatPumpTarget):
        raise ValueError(
            "with_heat_pump_streams takes a heat pump target, placed on the "
            f"process or on the utility system; this is a {target.type} target."
        )
    # The value the target was solved with: a per-call
    # ``minimum_approach_temperature`` or a zone's own setting can differ
    # from the problem's.
    return _case_with_heat_pump_streams(
        problem,
        target,
        dt_cont=float(target.provenance.effective_settings["HPR_DT_CONT"]),
        zone=zone,
        machine=machine,
        project_name=project_name,
    )


def _tree_node(tree: dict, path: list[str]) -> dict:
    """The node of the zone tree at ``path``, zone names from the root down."""
    node = tree
    for name in path[1:]:
        node = next(child for child in node["children"] if child["name"] == name)
    return node


def _require_heat_pump_site_target(case, target) -> None:
    """Raise unless ``case`` targets as the utility heat pump ``target`` did.

    The case holds the heat pump as streams that the site's utility levels
    serve. Its hot utility has to be what the target left, and its cold
    utility what the target left plus the heat the target let ambient air
    take, which the case does not hold. They differ when a level cannot
    exchange with the heat pump as the target assumed; the case would then
    look complete and target wrongly. The check runs on a copy, so the
    case stays unsolved.
    """
    copy = type(case)(case.to_problem_json(), project_name=case.project_name)
    site = copy.target.indirect_heat_integration(zone=target.provenance.zone_address)
    wanted = (
        float(target.hot_utility_target),
        float(target.cold_utility_target) + float(target.hpr_ambient_cold),
    )
    found = (float(site.hot_utility_target), float(site.cold_utility_target))
    tolerance = 1e-6 * max(1.0, *wanted)
    if all(abs(have - want) <= tolerance for have, want in zip(found, wanted)):
        return
    raise ValueError(
        "The site's utility levels cannot exchange with the heat pump as its "
        f"target assumed: with the heat pump's zones the site needs "
        f"{found[0]:.6g} kW of hot utility and {found[1]:.6g} kW of cold, "
        f"where the target leaves {wanted[0]:.6g} and {wanted[1]:.6g} kW. A "
        "level the heat pump delivers into must take heat as well as give "
        "it, and one it draws on must give heat as well as take it (type "
        "Both); a stream part way through a wide level's temperatures, and "
        "ambient air the evaporator draws on, are not built."
    )


def _band_move(stream, role: str, approach: float, levels) -> float:
    """The kelvin to move a utility heat pump stream by so that the
    utility level whose band it lies in serves all of it.

    A utility level is written as a narrow band of temperature (a steam
    main as 0.1 K), and a heat pump target places its streams against the
    heat in the band, so a stream can end inside it. A level serves a
    stream whole or not at all, and in the plant it is at one temperature.
    A condenser whose hot end, less the approaches, lies inside the band
    of a level that takes heat is moved up to the band's top; an
    evaporator whose cold end, plus the approaches, lies inside the band
    of a level that gives heat is moved down to its bottom. A segmented
    stream moves by the most any of its segments needs.
    """
    moves = [0.0]
    for level in levels:
        ends = sorted(
            float(end.value)
            for end in (level.supply_temperature, level.target_temperature)
        )
        if not 0.0 < ends[1] - ends[0] <= UTILITY_BAND_MAX_SPAN_K:
            continue
        gap = approach + float(level.delta_t_contribution.value)
        for part in stream.segments or (stream,):
            end = float(part.supply_temperature.value)
            if role == "condenser" and ends[0] + gap <= end < ends[1] + gap:
                moves.append(ends[1] + gap - end)
            if role == "evaporator" and ends[0] - gap < end <= ends[1] - gap:
                moves.append(ends[0] - gap - end)
    return max(moves, key=abs)


def _moved_record(record: dict, move: float) -> dict:
    """A stream record with its temperatures, and its segments', moved by
    ``move`` K."""

    def moved(part: dict) -> dict:
        return part | {
            key: part[key] | {"value": part[key]["value"] + move}
            for key in ("t_supply", "t_target")
        }

    if "segments" in record:
        return record | {"segments": [moved(part) for part in record["segments"]]}
    return moved(record)


def _case_with_utility_heat_pump_zones(
    problem, target, *, dt_cont: float, machine="HP", project_name=None
):
    """``problem`` with a utility heat pump ``target``'s streams, each side
    in a process zone of its own under the zone the target was solved on.

    The condensers are the hot streams of ``"<machine> condenser"`` and
    the evaporators the cold streams of ``"<machine> evaporator"``. In one
    zone the two would recover heat from each other.
    """
    root = problem._require_prepared_root_zone()
    scope = next(
        zone
        for zone in problem._walk_zone_tree(root)
        if zone.address == target.provenance.zone_address
    )
    if scope.type == ZoneType.P.value:
        raise ValueError(
            f"The heat pump was targeted on the process zone {scope.address!r}. "
            "A heat pump on the utility system is added as process zones of "
            "a site; target it on the site that holds the process."
        )
    source = problem.to_problem_json()
    node = _tree_node(source["zone_tree"], _ancestry(scope))
    approach = float(dt_cont)
    dt_cont = {"value": approach, "unit": "delta_degC"}
    for role, streams, levels in (
        ("condenser", target.hpr_hot_streams, scope.cold_utilities),
        ("evaporator", target.hpr_cold_streams, scope.hot_utilities),
    ):
        name = f"{machine} {role}"
        if any(child["name"] == name for child in node["children"] or ()):
            raise ValueError(
                f"Zone {scope.address!r} already has a zone named {name!r}, "
                "the name of a heat pump zone; give the heat pump another "
                "name with machine=."
            )
        node["children"] = [
            *(node["children"] or ()),
            {"name": name, "type": ZoneType.P.value, "children": None},
        ]
        for _key, stream in streams.items():
            record = _heat_pump_stream_record(
                stream,
                zone=f"{scope.address}/{name}",
                dt_cont=dt_cont,
                mark={"machine": machine, "role": role, "placement": "utility"},
            )
            source["streams"].append(
                _moved_record(record, _band_move(stream, role, approach, levels))
            )
    return type(problem)(
        source,
        project_name=problem.project_name if project_name is None else project_name,
    )


def _period_heat_pump_streams(target, periods) -> dict[str, dict[str, list]]:
    """A shared-design target's streams in each of ``periods``, by role and
    stream name.

    The target's own stream collections hold one period; its period
    outputs hold every period's.
    """
    outputs = target.hpr_details.period_outputs
    # A period the machine is off in (an off season, a load too small to
    # run for) is left out of the outputs, or is there without streams.
    each_output = [outputs.get(period) or {} for period in periods]
    if any(float(output.get("Q_amb_hot") or 0.0) > 0.0 for output in each_output):
        raise ValueError(
            "The heat pump draws on ambient air, and the air's stream is not "
            "built for a problem with several periods yet."
        )
    found = {}
    for role, group in (
        ("condenser", "hpr_hot_streams"),
        ("evaporator", "hpr_cold_streams"),
    ):
        each = [dict((output.get(group) or {}).items()) for output in each_output]
        names = list(dict.fromkeys(name for streams in each for name in streams))
        # ``None`` where the stream is off in a period.
        found[role] = {name: [streams.get(name) for streams in each] for name in names}
        for name, streams in found[role].items():
            if any(stream is not None and stream.has_segments for stream in streams):
                raise ValueError(
                    f"The heat pump stream {name!r} has segments, and a "
                    "segmented stream is not built for a problem with several "
                    "periods yet."
                )
    return found


def _period_thermal_record(streams) -> dict:
    """The temperatures and duty of one stream in each period, in order.

    Where the stream is off in a period (``None``) it has no duty there,
    at the temperatures of the first period it runs in: a stream needs a
    value in every period, and one with no heat flow takes no part.
    """
    running = next(stream for stream in streams if stream is not None)

    def values(attribute: str, off=None) -> dict:
        per_period = [
            _value_record(getattr(running if s is None else s, attribute))
            for s in streams
        ]
        return {
            "values": [
                record["value"] if s is not None or off is None else off
                for s, record in zip(streams, per_period)
            ],
            "unit": per_period[0]["unit"],
        }

    return {
        "t_supply": values("supply_temperature"),
        "t_target": values("target_temperature"),
        "heat_flow": values("heat_flow", off=0.0),
    }


def _require_heat_pump_stream_support(problem, target=None) -> None:
    """Raise for a problem, or a problem and target, heat pump streams
    cannot be built for.

    A problem with several periods needs a target solved as a shared
    design (``HPR_MULTIPERIOD_OPTIMIZATION_ENABLED``), which holds the
    heat pump's streams in every period; any other target holds one
    period's. Without ``target`` only the problem is checked.
    """
    if problem._process_components:
        raise ValueError(
            "heat pump streams cannot be added to a problem with process "
            "components: the new case is built from the input records, which "
            "do not hold them."
        )
    if (
        target is not None
        and len(problem.period_ids) > 1
        and not getattr(target, "hpr_shared_design", False)
    ):
        raise ValueError(
            f"The problem has {len(problem.period_ids)} periods and the heat "
            "pump target holds one period's streams. Solve it as a shared "
            "design, with the option HPR_MULTIPERIOD_OPTIMIZATION_ENABLED, or "
            "derive a single-period case and target the heat pump on it."
        )


def _case_with_heat_pump_streams(
    problem, target, *, dt_cont: float, zone=None, machine="HP", project_name=None
):
    """``problem`` with the streams of ``target`` at ``dt_cont`` (K), with no
    check that the target is this problem's own."""
    _require_heat_pump_stream_support(problem, target)
    address = _heat_pump_zone_address(problem, target, zone)
    dt_cont = {"value": float(dt_cont), "unit": "delta_degC"}
    source = problem.to_problem_json()
    taken = {
        record["name"] for record in source["streams"] if record["zone"] == address
    }
    # The air an evaporator draws on, at the air's own contribution
    # (HPR_DT_ENV_CONT). The air a condenser rejects to is left out: the
    # case's cold utility takes that heat.
    air = [
        stream
        for _key, stream in target.hpr_details.amb_streams.items()
        if stream.stream_type == StreamType.Hot.value
    ]
    periods = list(problem.period_ids)
    by_period = _period_heat_pump_streams(target, periods) if len(periods) > 1 else {}
    groups = {
        "condenser": [s for _key, s in target.hpr_hot_streams.items()],
        "evaporator": [s for _key, s in target.hpr_cold_streams.items()],
        "ambient_source": air,
    }
    if by_period:
        # A shared design: the target's own collections hold one period,
        # which may be one the machine is off in. Each stream is taken
        # from the first period it runs in. Air over several periods is
        # refused where the period outputs are read.
        groups = {
            role: [next(s for s in each if s is not None) for each in named.values()]
            for role, named in by_period.items()
        }
    for role, streams in groups.items():
        for stream in streams:
            if stream.name in taken:
                raise ValueError(
                    f"Zone {address!r} already has a stream named "
                    f"{stream.name!r}, the name of a heat pump stream; rename "
                    "the process stream."
                )
            record = _heat_pump_stream_record(
                stream,
                zone=address,
                dt_cont=(
                    _value_record(stream.delta_t_contribution)
                    if role == "ambient_source"
                    else dt_cont
                ),
                mark={"machine": machine, "role": role},
            )
            if by_period:
                # A shared design: the stream's value in every period.
                record |= _period_thermal_record(by_period[role][stream.name])
            source["streams"].append(record)
    return type(problem)(
        source,
        project_name=problem.project_name if project_name is None else project_name,
    )


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
