"""Application-owned case derivation with independent runtime ownership."""

from copy import deepcopy

from ._problem.input.utilities import _set_utilities_for_zone_and_subzones
from ._problem.targeting.state import snapshot_problem


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
    from ..contracts.units import INPUT_UNIT_RULES, standardise_input_value

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
            source = standardise_input_value(
                item, field_name=field, config=donor_config
            )
            local = standardise_input_value(
                item, field_name=field, config=receiver_config
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
