"""Deterministic no-solver executor for HEN workflow tests."""

from __future__ import annotations

from collections.abc import Sequence

from .....classes.heat_exchanger import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerStreamRole,
)
from .....classes.heat_exchanger_network import HeatExchangerNetwork
from .....lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
    SynthesisMethod,
)
from ..errors import WorkflowContractError


class FakeSynthesisExecutor:
    """Deterministic no-solver executor for workflow and cache tests."""

    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = set(failures or ())
        self.stage_order: list[SynthesisMethod] = []
        self.executed_tasks: list[HeatExchangerNetworkSynthesisTask] = []

    def execute(
        self,
        tasks: Sequence[HeatExchangerNetworkSynthesisTask],
        *,
        problem,
        parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
        max_parallel: int,
    ) -> tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]:
        if tasks:
            self.stage_order.append(tasks[0].method)

        outcomes: list[HeatExchangerNetworkSynthesisTaskOutcome] = []
        for task in tasks:
            if task.parent_task_id is not None:
                parent = parent_outcomes.get(task.parent_task_id)
                if parent is None or parent.status != "success":
                    raise WorkflowContractError(
                        "downstream heat exchanger network tasks require "
                        "successful parent outcomes"
                    )

            self.executed_tasks.append(task)
            if task.task_id in self.failures:
                outcomes.append(
                    HeatExchangerNetworkSynthesisTaskOutcome(
                        task=task,
                        status="failed",
                        solver_status="fake-failed",
                        error="configured fake executor failure",
                    )
                )
                continue

            network = _fake_network(problem, task)
            outcomes.append(
                HeatExchangerNetworkSynthesisTaskOutcome(
                    task=task,
                    status="success",
                    network=network,
                    objective_value=network.total_annual_cost,
                    solver_status="fake-optimal",
                )
            )
        return tuple(outcomes)


def _fake_network(
    problem,
    task: HeatExchangerNetworkSynthesisTask,
) -> HeatExchangerNetwork:
    zone = problem.master_zone
    if zone is None:
        raise RuntimeError(
            "Fake heat exchanger network execution requires a prepared root Zone."
        )

    hot_key, hot_stream = _first_stream(zone.hot_streams.items(), "hot process")
    cold_key, cold_stream = _first_stream(zone.cold_streams.items(), "cold process")
    hot_utility_key, _hot_utility = _first_stream(
        zone.hot_utilities.items(),
        "hot utility",
    )
    cold_utility_key, _cold_utility = _first_stream(
        zone.cold_utilities.items(),
        "cold utility",
    )
    stage_count = task.stage_count or int(zone.config.hens.stage_selection[0])
    duty = _fake_duty(task)
    utility_cost = round(duty * 0.35, 6)
    capital_cost = round(duty * 1.15, 6)
    total_annual_cost = round(utility_cost + capital_cost, 6)

    exchangers = (
        HeatExchanger(
            exchanger_id=f"{task.task_id}-recovery",
            kind=HeatExchangerKind.RECOVERY,
            source_stream=hot_key,
            sink_stream=cold_key,
            source_stream_role=HeatExchangerStreamRole.PROCESS,
            sink_stream_role=HeatExchangerStreamRole.PROCESS,
            stage=stage_count,
            duty=duty,
            area=round(duty / 10.0, 6),
            approach_temperatures=(task.approach_temperature,),
            source_inlet_temperature=_temperature(hot_stream.t_supply, "K"),
            source_outlet_temperature=_temperature(hot_stream.t_target, "K"),
            sink_inlet_temperature=_temperature(cold_stream.t_supply, "K"),
            sink_outlet_temperature=_temperature(cold_stream.t_target, "K"),
            capital_cost=capital_cost,
            total_annual_cost=total_annual_cost,
        ),
        HeatExchanger(
            exchanger_id=f"{task.task_id}-hot-utility",
            kind=HeatExchangerKind.HOT_UTILITY,
            source_stream=hot_utility_key,
            sink_stream=cold_key,
            source_stream_role=HeatExchangerStreamRole.UTILITY,
            sink_stream_role=HeatExchangerStreamRole.PROCESS,
            duty=round(duty * 0.1, 6),
            operating_cost=round(utility_cost * 0.5, 6),
        ),
        HeatExchanger(
            exchanger_id=f"{task.task_id}-cold-utility",
            kind=HeatExchangerKind.COLD_UTILITY,
            source_stream=hot_key,
            sink_stream=cold_utility_key,
            source_stream_role=HeatExchangerStreamRole.PROCESS,
            sink_stream_role=HeatExchangerStreamRole.UTILITY,
            duty=round(duty * 0.08, 6),
            operating_cost=round(utility_cost * 0.5, 6),
        ),
    )
    return HeatExchangerNetwork(
        exchangers=exchangers,
        run_id=task.run_id,
        task_id=task.task_id,
        state_id=task.state_id,
        method=task.method,
        stage_count=stage_count,
        objective_value=total_annual_cost,
        total_annual_cost=total_annual_cost,
        utility_cost=utility_cost,
        capital_cost=capital_cost,
        summary_metrics={
            "recovery_unit_count": 1,
            "hot_utility_unit_count": 1,
            "cold_utility_unit_count": 1,
            "approach_temperature": task.approach_temperature,
            "derivative_threshold": task.derivative_threshold,
        },
    )


def _first_stream(items, label: str):
    try:
        return next(iter(items))
    except StopIteration as exc:
        raise ValueError(
            f"heat exchanger network synthesis requires at least one {label} stream."
        ) from exc


def _fake_duty(task: HeatExchangerNetworkSynthesisTask) -> float:
    derivative = 0.0 if task.derivative_threshold is None else task.derivative_threshold
    stage = 0 if task.stage_count is None else task.stage_count
    return round(
        1000.0 + task.approach_temperature * 10.0 + derivative * 100.0 + stage,
        6,
    )


def _temperature(value, unit: str) -> float | None:
    if value is None:
        return None
    if hasattr(value, "to"):
        return float(value.to(unit).value)
    return float(value)


__all__ = ["FakeSynthesisExecutor"]
