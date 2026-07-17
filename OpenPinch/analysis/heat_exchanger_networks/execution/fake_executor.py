"""Deterministic no-solver executor for HEN workflow tests."""

from __future__ import annotations

from collections.abc import Sequence

from ....contracts.synthesis.common import SynthesisMethod
from ....contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ....domain._heat_exchanger.period_state import HeatExchangerPeriodState
from ....domain.enums import HeatExchangerKind, StreamID
from ....domain.heat_exchanger import HeatExchanger
from ....domain.heat_exchanger_network import HeatExchangerNetwork
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
    approach_temperature = max(float(task.approach_temperature), 1.0)
    temperatures = _fake_temperatures(
        hot_supply=_temperature(hot_stream.t_supply, "K"),
        cold_supply=_temperature(cold_stream.t_supply, "K"),
        approach_temperature=approach_temperature,
    )
    hot_utility_duty = round(duty * 0.1, 6)
    cold_utility_duty = round(duty * 0.08, 6)
    period_id = task.period_id or "0"

    exchangers = (
        HeatExchanger(
            exchanger_id=f"{task.task_id}-recovery",
            kind=HeatExchangerKind.RECOVERY,
            source_stream=hot_key,
            sink_stream=cold_key,
            source_stream_role=StreamID.Process,
            sink_stream_role=StreamID.Process,
            stage=stage_count,
            period_states=(
                HeatExchangerPeriodState(
                    period_id=period_id,
                    period_idx=0,
                    duty=duty,
                    approach_temperatures=(
                        temperatures["source_inlet"] - temperatures["sink_outlet"],
                        temperatures["source_outlet"] - temperatures["sink_inlet"],
                    ),
                    source_inlet_temperature=temperatures["source_inlet"],
                    source_outlet_temperature=temperatures["source_outlet"],
                    sink_inlet_temperature=temperatures["sink_inlet"],
                    sink_outlet_temperature=temperatures["sink_outlet"],
                ),
            ),
            area=round(duty / 10.0, 6),
            capital_cost=capital_cost,
        ),
        HeatExchanger(
            exchanger_id=f"{task.task_id}-hot-utility",
            kind=HeatExchangerKind.HOT_UTILITY,
            source_stream=hot_utility_key,
            sink_stream=cold_key,
            source_stream_role=StreamID.Utility,
            sink_stream_role=StreamID.Process,
            period_states=(
                HeatExchangerPeriodState(
                    period_id=period_id,
                    period_idx=0,
                    duty=hot_utility_duty,
                    source_inlet_temperature=temperatures["hot_utility_inlet"],
                    source_outlet_temperature=temperatures["hot_utility_outlet"],
                    sink_inlet_temperature=temperatures["sink_outlet"],
                    sink_outlet_temperature=temperatures["sink_outlet"] + 10.0,
                ),
            ),
            area=round(hot_utility_duty / 10.0, 6),
        ),
        HeatExchanger(
            exchanger_id=f"{task.task_id}-cold-utility",
            kind=HeatExchangerKind.COLD_UTILITY,
            source_stream=hot_key,
            sink_stream=cold_utility_key,
            source_stream_role=StreamID.Process,
            sink_stream_role=StreamID.Utility,
            period_states=(
                HeatExchangerPeriodState(
                    period_id=period_id,
                    period_idx=0,
                    duty=cold_utility_duty,
                    source_inlet_temperature=temperatures["source_outlet"],
                    source_outlet_temperature=temperatures["source_outlet"] - 10.0,
                    sink_inlet_temperature=temperatures["cold_utility_inlet"],
                    sink_outlet_temperature=temperatures["cold_utility_outlet"],
                ),
            ),
            area=round(cold_utility_duty / 10.0, 6),
        ),
    )
    return HeatExchangerNetwork(
        exchangers=exchangers,
        run_id=task.run_id,
        task_id=task.task_id,
        period_id=task.period_id,
        method=task.method,
        stage_count=stage_count,
        objective_value=total_annual_cost,
        total_annual_cost=total_annual_cost,
        utility_cost=utility_cost,
        capital_cost=capital_cost,
        summary_metrics={
            "total_units": 3,
            "recovery_units": 1,
            "hot_utility_units": 1,
            "cold_utility_units": 1,
            "recovery_load": duty,
            "hot_utility_load": hot_utility_duty,
            "cold_utility_load": cold_utility_duty,
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


def _fake_temperatures(
    *,
    hot_supply: float | None,
    cold_supply: float | None,
    approach_temperature: float,
) -> dict[str, float]:
    sink_inlet = cold_supply if cold_supply is not None else 300.0
    source_inlet = hot_supply if hot_supply is not None else sink_inlet + 200.0
    source_inlet = max(source_inlet, sink_inlet + approach_temperature + 100.0)
    sink_outlet = min(sink_inlet + 50.0, source_inlet - approach_temperature - 10.0)
    source_outlet = max(
        sink_inlet + approach_temperature + 10.0,
        source_inlet - 80.0,
    )
    source_outlet = min(source_outlet, source_inlet - approach_temperature - 10.0)
    return {
        "source_inlet": source_inlet,
        "source_outlet": source_outlet,
        "sink_inlet": sink_inlet,
        "sink_outlet": sink_outlet,
        "hot_utility_inlet": sink_outlet + approach_temperature + 50.0,
        "hot_utility_outlet": sink_outlet + approach_temperature + 40.0,
        "cold_utility_inlet": sink_inlet - 30.0,
        "cold_utility_outlet": sink_inlet - 20.0,
    }


__all__ = ["FakeSynthesisExecutor"]
