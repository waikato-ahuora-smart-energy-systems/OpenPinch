"""Internal quality-tier pathway planning for OpenHENS workflows.

Tier semantics:

* 0: fastest compact PDM directly to EVM.
* 1: exact standard OpenHENS PDM -> TDM -> EVM route.
* 2: protected tiers 0/1 plus base-dTmin compact/direct and raw/TDM routes.
* 3: base and doubled-dTmin compact/direct and raw/TDM routes.
* 4: tier-3 dTmin routes with wider EVM branch exploration.
* 5: experimental reduced/base/doubled-dTmin search at tier-4 breadth.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

PDMMode = Literal["compact", "raw", "standard"]

_TIER_MULTIPLIERS: dict[int, tuple[float, ...]] = {
    2: (1.0,),
    3: (1.0, 2.0),
    4: (1.0, 2.0),
    5: (0.5, 1.0, 2.0),
}

_TIER_EVM_BREADTH: dict[int, int] = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
}


@dataclass(frozen=True)
class TierPathway:
    """One protected or expanded route through the OpenHENS workflow."""

    pathway_id: str
    tier_origin: int
    pathway_kind: str
    pdm_mode: PDMMode
    multiplier: float | None
    uses_tdm: bool
    evm_n_ad_branches: int
    evm_n_rm_branches: int
    evm_no_improvement_patience: int | None
    protected: bool
    exact_open_hens: bool = False

    def metadata(self) -> dict[str, object]:
        return {
            "pathway_id": self.pathway_id,
            "tier_origin": self.tier_origin,
            "pathway_kind": self.pathway_kind,
            "pdm_mode": self.pdm_mode,
            "pdm_multiplier": self.multiplier,
            "uses_tdm": self.uses_tdm,
            "evm_n_ad_branches": self.evm_n_ad_branches,
            "evm_n_rm_branches": self.evm_n_rm_branches,
            "evm_no_improvement_patience": self.evm_no_improvement_patience,
            "protected": self.protected,
            "exact_open_hens": self.exact_open_hens,
        }


def tier_pdm_multipliers(tier: int) -> tuple[float, ...]:
    """Return tier-generated dt-cont multipliers for expanded PDM paths."""

    return _TIER_MULTIPLIERS.get(int(tier), ())


def tier_evm_branch_breadth(tier: int) -> int:
    """Return default add/remove EVM branch breadth for one tier."""

    return _TIER_EVM_BREADTH.get(int(tier), 3)


def tier_pathways(settings) -> tuple[TierPathway, ...]:
    """Return the monotonic pathway set for the configured quality tier."""

    tier = int(settings.synthesis_quality_tier)
    if tier == 0:
        return tuple(_tier0_pathways(settings, protected=False))
    if tier == 1:
        return (_tier1_pathway(settings, protected=False),)

    pathways: list[TierPathway] = []
    pathways.extend(_tier0_pathways(settings, protected=True))
    pathways.append(_tier1_pathway(settings, protected=True))
    for origin in range(2, tier + 1):
        pathways.extend(_quality_tier_pathways(settings, origin))
    return tuple(pathways)


def pathways_from_metadata(metadata: dict) -> tuple[TierPathway, ...]:
    """Rehydrate pathway specs stored on task metadata."""

    raw_pathways = metadata.get("pathways")
    if not isinstance(raw_pathways, Iterable) or isinstance(raw_pathways, (str, bytes)):
        return ()
    pathways = []
    for raw in raw_pathways:
        if not isinstance(raw, dict):
            continue
        pathways.append(
            TierPathway(
                pathway_id=str(raw["pathway_id"]),
                tier_origin=int(raw["tier_origin"]),
                pathway_kind=str(raw["pathway_kind"]),
                pdm_mode=str(raw["pdm_mode"]),  # type: ignore[arg-type]
                multiplier=(
                    None
                    if raw.get("pdm_multiplier") is None
                    else float(raw["pdm_multiplier"])
                ),
                uses_tdm=bool(raw["uses_tdm"]),
                evm_n_ad_branches=int(raw["evm_n_ad_branches"]),
                evm_n_rm_branches=int(raw["evm_n_rm_branches"]),
                evm_no_improvement_patience=(
                    None
                    if raw.get("evm_no_improvement_patience") is None
                    else int(raw["evm_no_improvement_patience"])
                ),
                protected=bool(raw["protected"]),
                exact_open_hens=bool(raw.get("exact_open_hens", False)),
            )
        )
    return tuple(pathways)


def pathway_metadata(pathways: Iterable[TierPathway]) -> dict[str, object]:
    """Return compact task metadata for one or more pathway specs."""

    ordered = tuple(pathways)
    if not ordered:
        return {}
    first = ordered[0]
    return {
        "pathways": [pathway.metadata() for pathway in ordered],
        "pathway_ids": [pathway.pathway_id for pathway in ordered],
        "tier_origin": first.tier_origin,
        "pathway_kind": first.pathway_kind,
        "pdm_mode": first.pdm_mode,
        "protected_pathway": any(pathway.protected for pathway in ordered),
    }


def _tier0_pathways(settings, *, protected: bool) -> tuple[TierPathway, ...]:
    multipliers = _explicit_or_default_multipliers(settings, default=(1.0,))
    return tuple(
        TierPathway(
            pathway_id=f"tier0-compact-{_multiplier_label(multiplier)}",
            tier_origin=0,
            pathway_kind="tier0_pdm_evm",
            pdm_mode="compact",
            multiplier=float(multiplier),
            uses_tdm=False,
            evm_n_ad_branches=1,
            evm_n_rm_branches=1,
            evm_no_improvement_patience=None,
            protected=protected,
        )
        for multiplier in multipliers
    )


def _tier1_pathway(settings, *, protected: bool) -> TierPathway:
    return TierPathway(
        pathway_id="tier1-open-hens",
        tier_origin=1,
        pathway_kind="tier1_open_hens",
        pdm_mode="standard",
        multiplier=None,
        uses_tdm=True,
        evm_n_ad_branches=1,
        evm_n_rm_branches=1,
        evm_no_improvement_patience=None,
        protected=protected,
        exact_open_hens=True,
    )


def _quality_tier_pathways(settings, tier: int) -> tuple[TierPathway, ...]:
    multipliers = _explicit_or_default_multipliers(
        settings,
        default=tier_pdm_multipliers(tier),
    )
    ad_branches = _branch_breadth(
        settings.evm_n_ad_branches,
        tier=tier,
    )
    rm_branches = _branch_breadth(
        settings.evm_n_rm_branches,
        tier=tier,
    )
    patience = 2 if max(ad_branches, rm_branches) > 1 else None
    pathways: list[TierPathway] = []
    for multiplier in multipliers:
        for mode in ("compact", "raw"):
            pathways.append(
                TierPathway(
                    pathway_id=(f"tier{tier}-{mode}-{_multiplier_label(multiplier)}"),
                    tier_origin=tier,
                    pathway_kind=f"tier{tier}_quality",
                    pdm_mode=mode,  # type: ignore[arg-type]
                    multiplier=float(multiplier),
                    uses_tdm=mode == "raw",
                    evm_n_ad_branches=ad_branches,
                    evm_n_rm_branches=rm_branches,
                    evm_no_improvement_patience=patience,
                    protected=False,
                )
            )
    return tuple(pathways)


def _explicit_or_default_multipliers(
    settings,
    *,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    if settings.user_dt_cont_multipliers:
        return tuple(float(value) for value in settings.dt_cont_multipliers or ())
    return default


def _branch_breadth(value: int | None, *, tier: int) -> int:
    if value is not None:
        return max(1, int(value))
    return tier_evm_branch_breadth(tier)


def _multiplier_label(value: float) -> str:
    return ("%g" % float(value)).replace(".", "p")


__all__ = [
    "TierPathway",
    "pathway_metadata",
    "pathways_from_metadata",
    "tier_evm_branch_breadth",
    "tier_pathways",
    "tier_pdm_multipliers",
]
