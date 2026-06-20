"""Validation shell for accepted heat exchanger network synthesis design payloads."""

from __future__ import annotations

from ....lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult


def verify_synthesis_result(
    result: HeatExchangerNetworkSynthesisResult,
) -> tuple[str, ...]:
    """Return contract failures for an accepted design result."""
    failures: list[str] = []
    if not result.network.exchangers:
        failures.append(
            "accepted heat exchanger network design must contain at least one exchanger"
        )
    if result.network.run_id != result.run_id:
        failures.append("network run_id must match the enclosing design result")
    if result.task_id is not None and result.network.task_id != result.task_id:
        failures.append("network task_id must match the accepted task id")
    if (
        result.network.total_annual_cost is None
        and "total_annual_cost" not in result.objective_values
    ):
        failures.append(
            "accepted heat exchanger network design must include total annual "
            "cost metadata"
        )
    for outcome in result.ranked_networks:
        if outcome.status == "success" and outcome.network is None:
            failures.append(
                f"successful task {outcome.task.task_id} is missing network output"
            )
    return tuple(failures)
