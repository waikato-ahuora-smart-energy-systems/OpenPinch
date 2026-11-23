"""The cost a heat pump adds to a network designed around its streams."""

from __future__ import annotations

from collections.abc import Sequence

from ...domain.heat_exchanger_network import HeatExchangerNetwork, NetworkHeatPump
from ..economics import compute_capital_recovery_factor


def network_with_heat_pumps(
    network: HeatExchangerNetwork, heat_pumps: Sequence[NetworkHeatPump]
) -> HeatExchangerNetwork:
    """``network`` with ``heat_pumps`` recorded on it and their annual cost
    (electricity and annualised capital) in its total.

    The utility cost and the capital cost stay the network's own: its
    utilities and its exchangers.
    """
    machines = tuple(heat_pumps)
    total = network.total_annual_cost
    if total is not None:
        total += sum(machine.annual_cost for machine in machines)
    return network.model_copy(
        update={"heat_pumps": machines, "total_annual_cost": total}
    )


def heat_pump_record(
    target, *, machine: str, costing, cost_per_kw: float, period_weights=None
) -> NetworkHeatPump:
    """A heat pump target as the machine a network records.

    Electricity is the target's work at ``COSTING_HPR_ELE_PRICE`` over
    ``COSTING_ANNUAL_OP_TIME``. Capital is ``cost_per_kw`` ($ per kW of
    condenser capacity) for the compressor and its motor, annualised at
    the capital recovery factor of ``COSTING_DISCOUNT_RATE`` and
    ``COSTING_SERVICE_LIFE``. The condenser and evaporator are exchangers
    of the network, priced there.
    """
    work = float(target.hpr_work)
    condenser = _duty(target.hpr_hot_streams)
    evaporator = _duty(target.hpr_cold_streams)
    cop = float(target.hpr_cop)
    outputs = getattr(getattr(target, "hpr_details", None), "period_outputs", None)
    if period_weights and outputs:
        # A shared design over several periods: the target's own fields
        # hold one period, its period outputs every one.
        total = float(sum(period_weights.values()))
        share = {period: weight / total for period, weight in period_weights.items()}
        delivered = {p: _duty(outputs[p]["hpr_hot_streams"]) for p in share}
        work = sum(share[p] * float(outputs[p]["w_net"]) for p in share)
        cop = sum(share[p] * delivered[p] for p in share) / work if work else cop
        condenser = max(delivered.values())
        evaporator = max(_duty(outputs[p]["hpr_cold_streams"]) for p in share)
    capital = float(cost_per_kw) * condenser
    return NetworkHeatPump(
        machine=machine,
        cycle=str(target.hpr_cycle),
        condenser_duty=condenser,
        evaporator_duty=evaporator,
        work=work,
        cop=cop,
        electricity_cost=(
            work * float(costing.hpr_ele_price) * float(costing.annual_op_time) / 1e3
        ),
        capital_cost=capital,
        annual_capital_cost=capital
        * compute_capital_recovery_factor(costing.discount_rate, costing.service_life),
        load_fraction=_load_fraction(getattr(target, "hpr_load", None)),
    )


def _load_fraction(load) -> float | None:
    """The share of the available heat pump load a target's summary says it
    took, or None without a summary or an available load."""
    if load is None or not float(load.available) > 0.0:
        return None
    return float(load.selected) / float(load.available)


def _duty(streams) -> float:
    """The summed duty (kW) of a target's condenser or evaporator streams."""
    return float(sum(float(stream.heat_flow.value) for _key, stream in streams.items()))


__all__ = ["heat_pump_record", "network_with_heat_pumps"]
