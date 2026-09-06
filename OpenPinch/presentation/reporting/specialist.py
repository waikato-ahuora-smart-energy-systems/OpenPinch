"""Report adapters preserving independent specialist result contracts."""

from ...contracts.heat_recovery_dt_min import HeatRecoveryDtMinResult
from ...contracts.hpr_performance_map import HprPerformanceMap
from ...contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ...contracts.utility_placement import UtilityPlacementResult


def specialist_result(target, *, is_total=False):
    """Specialist results already carry typed quantities and their own schema."""
    return target.model_copy(deep=True)


SPECIALIST_RESULT_ADAPTERS = {
    family: specialist_result
    for family in (
        HeatRecoveryDtMinResult,
        HprPerformanceMap,
        HeatExchangerNetworkSynthesisResult,
        UtilityPlacementResult,
    )
}
