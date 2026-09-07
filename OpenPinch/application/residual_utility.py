"""Detach and allocate a selected HPR residual using canonical problem input."""

import math

from ..analysis.targeting.residual import compute_residual_utility_target
from ..domain.enums import TargetType
from ..domain.hpr import HPRResidualSnapshot
from .hpr_selection import hpr_result_digest, validate_hpr_target


def residual_basis(problem):
    return (
        problem._validated_data.residual_basis
        if problem._validated_data is not None
        else None
    )


def create_hpr_residual_case(problem, *, base_target, project_name=None):
    target = validate_hpr_target(problem, base_target)
    data = target.hpr_residual
    if data is None or target.hpr_load is None:
        raise ValueError(
            "Residual utility conversion requires a supported scalar HPR target."
        )
    basis = HPRResidualSnapshot(
        data=data,
        source_identity=target.provenance.identity,
        source_fingerprint=target.provenance.input_fingerprint,
        source_zone=target.scope,
        result_digest=hpr_result_digest(target),
    )
    idx = target.period_idx

    def scalar(value):
        return (
            None
            if value is None
            else {"value": float(value[idx]), "unit": str(value.unit)}
        )

    def maximum_duty(value):
        # Identity-aware caps use NaN internally for periods without a limit.
        # Canonical scalar input expresses that case as an absent capacity.
        if value is None or math.isnan(float(value[idx])):
            return None
        return scalar(value)

    utilities = []
    for side, collection in (
        ("Hot", target.hot_utilities),
        ("Cold", target.cold_utilities),
    ):
        for u in collection:
            item = {
                "name": u.name,
                "type": side,
                "t_supply": scalar(u.supply_temperature),
                "t_target": scalar(u.target_temperature),
                "heat_flow": 0.0,
                "dt_cont": scalar(u.effective_delta_t_contribution),
                "htc": scalar(u.heat_transfer_coefficient),
                "price": scalar(u.price),
                "active": u.is_active,
                "maximum_heat_flow": maximum_duty(u.maximum_heat_flow),
                "fluid_name": u.fluid_name,
                "fluid_phase": u.fluid_phase,
                "p_supply": scalar(u.supply_pressure),
                "p_target": scalar(u.target_pressure),
                "h_supply": scalar(u.supply_enthalpy),
                "h_target": scalar(u.target_enthalpy),
            }
            if u.segments:
                item["heat_flow"] = None
                duties = [float(part.heat_flow[idx]) for part in u.segments]
                total = sum(duties)
                fractions = (
                    [q / total for q in duties]
                    if total > 0
                    else u._segment_targeting_fractions.get(idx)
                )
                if fractions is None:
                    raise ValueError("Segmented utility has no scalable duty profile.")
                item["segments"] = [
                    {
                        "t_supply": scalar(part.supply_temperature),
                        "t_target": scalar(part.target_temperature),
                        "heat_flow": {"value": fraction, "unit": "kW"},
                        "dt_cont": scalar(part.effective_delta_t_contribution),
                        "htc": scalar(part.heat_transfer_coefficient),
                        "price": scalar(part.price),
                        "p_supply": scalar(part.supply_pressure),
                        "p_target": scalar(part.target_pressure),
                        "h_supply": scalar(part.supply_enthalpy),
                        "h_target": scalar(part.target_enthalpy),
                    }
                    for part, fraction in zip(u.segments, fractions, strict=True)
                ]
            utilities.append(item)
    from ..domain.configuration_fields import USER_CONFIG_FIELD_SPECS

    options = {
        k: v
        for k, v in target.config._values.items()
        if k in USER_CONFIG_FIELD_SPECS and not k.startswith("HPR_")
    }
    options.update(
        PROBLEM_PERIOD_IDS=[data.period_id],
        PROBLEM_PERIOD_WEIGHTS=[1.0],
        HPR_MULTIPERIOD_OPTIMIZATION_ENABLED=False,
    )
    name = project_name or f"{problem.project_name} residual"
    return type(problem)(
        source={
            "streams": [],
            "utilities": utilities,
            "options": options,
            "residual_basis": basis.model_dump(mode="json"),
            "zone_tree": {"name": name, "type": "Process Zone"},
        },
        project_name=name,
    )


def residual_utility_service(data):
    def service(zone, args=None):
        if args and args.get("period_id") not in (None, data.period_id):
            raise ValueError("Select the frozen residual period.")
        target = compute_residual_utility_target(zone, data)
        zone.targets[TargetType.DI.value] = target
        return zone

    return service


def prepare_residual_problem(input_data, project_name):
    from ..domain.configuration import Configuration
    from ..domain.enums import ZoneType
    from ..domain.zone import Zone
    from ._problem.input.utilities import _get_hot_and_cold_utilities

    data = input_data.residual_basis.data
    config = Configuration(
        options=input_data.options or {},
        top_zone_name=project_name,
        top_zone_identifier=ZoneType.P.value,
    )
    zone = Zone(
        name=input_data.zone_tree.name
        if input_data.zone_tree is not None
        else project_name,
        type=ZoneType.P.value,
        config=config,
    )
    zone.set_period_context({data.period_id: 0}, config.problem.period_weights, 1)
    temperatures = data.profile.temperatures
    utilities = _get_hot_and_cold_utilities(
        utilities=input_data.utilities,
        hu_t_min=max(temperatures),
        cu_t_max=min(temperatures),
        config=config,
    )
    from ..analysis.targeting.utilities import _apply_utility_duties

    _apply_utility_duties(utilities, (0.0,) * len(utilities), idx=0)
    zone.hot_utilities = utilities.get_hot_utility_streams()
    zone.cold_utilities = utilities.get_cold_utility_streams()
    return zone
