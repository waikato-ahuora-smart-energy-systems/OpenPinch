"""Site utility system contracts for total site heat exchanger network design.

A site's utility levels are external hot utilities (bought), external cold
utilities (heat rejected at a price) and mains (``"both"``: steam headers or
hot water circuits that processes draw from and supply). Prices are per kW
in whatever time basis the caller uses (the design passes $/kW/y, so costs
are $/y), one value per period.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SiteUtilityLevel(BaseModel):
    """One utility level of a site.

    ``import_price`` is the price of buying heat into the level (hot and
    both levels); ``dump_price`` that of rejecting a surplus from it (cold
    and both levels). ``import_capacity`` bounds the duty bought per period
    (``None`` for no bound). ``t_high`` and ``t_low`` span the level's
    temperatures; heat is let down from one level to another only when the
    first lies wholly above the second.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: Literal["hot", "cold", "both"]
    t_high: float
    t_low: float
    import_price: tuple[float, ...] | None = None
    dump_price: tuple[float, ...] | None = None
    import_capacity: tuple[float | None, ...] | None = None

    @model_validator(mode="after")
    def _check(self) -> SiteUtilityLevel:
        if not (math.isfinite(self.t_high) and math.isfinite(self.t_low)):
            raise ValueError("level temperatures must be finite")
        if self.t_high < self.t_low:
            raise ValueError("t_high must not be below t_low")
        imports = self.kind in ("hot", "both")
        dumps = self.kind in ("cold", "both")
        if imports != (self.import_price is not None):
            raise ValueError(
                f"a {self.kind} level {'needs' if imports else 'takes no'} import_price"
            )
        if dumps != (self.dump_price is not None):
            raise ValueError(
                f"a {self.kind} level {'needs' if dumps else 'takes no'} dump_price"
            )
        if self.import_capacity is not None and not imports:
            raise ValueError("only a hot or both level takes import_capacity")
        for prices in (self.import_price, self.dump_price):
            if prices is not None and not all(math.isfinite(p) for p in prices):
                raise ValueError("level prices must be finite")
        return self

    @property
    def imports(self) -> bool:
        return self.import_price is not None

    @property
    def dumps(self) -> bool:
        return self.dump_price is not None


class SiteUtilityLevels(BaseModel):
    """A site's utility levels, its period weights and the letdown price."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    levels: tuple[SiteUtilityLevel, ...] = Field(min_length=1)
    period_weights: tuple[float, ...] = (1.0,)
    letdown_price: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> SiteUtilityLevels:
        names = [level.name for level in self.levels]
        if len(set(names)) != len(names):
            raise ValueError("level names must be unique")
        periods = len(self.period_weights)
        if periods < 1 or any(w < 0.0 for w in self.period_weights):
            raise ValueError("period weights must be non-negative")
        if sum(self.period_weights) <= 0.0:
            raise ValueError("period weights must not all be zero")
        for level in self.levels:
            for values in (
                level.import_price,
                level.dump_price,
                level.import_capacity,
            ):
                if values is not None and len(values) != periods:
                    raise ValueError(
                        f"level {level.name} needs one value per period ({periods})"
                    )
        return self

    def level(self, name: str) -> SiteUtilityLevel:
        for level in self.levels:
            if level.name == name:
                return level
        raise KeyError(name)


class LevelLoad(BaseModel):
    """A process's duty per period on one level: heaters (use) and coolers
    (generation), in kW."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    use: tuple[float, ...]
    generation: tuple[float, ...]

    @model_validator(mode="after")
    def _check(self) -> LevelLoad:
        if len(self.use) != len(self.generation):
            raise ValueError("use and generation need one value per period each")
        if any(not math.isfinite(q) or q < 0.0 for q in self.use + self.generation):
            raise ValueError("level loads must be finite and non-negative")
        return self


class UtilityLevelBalance(BaseModel):
    """One level's balance in one period.

    ``use + letdown_out + dump == generation + letdown_in + imported``.
    ``cost`` is what the level's import, dump and letdown in cost.
    ``marginal_price`` is the dual of the balance: the cost of one more kW
    of use on the level (so one more kW of generation saves it).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: str
    period_idx: int
    use: float
    generation: float
    imported: float
    letdown_in: float
    letdown_out: float
    dump: float
    cost: float
    marginal_price: float


class UtilitySystemBalance(BaseModel):
    """The site utility balance of every level and period, with totals.

    ``operating_cost`` is the period-weighted mean of ``period_costs``;
    ``hot_utility`` and ``cold_utility`` are the site's imports and dumps
    per period.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    balances: tuple[UtilityLevelBalance, ...]
    period_costs: tuple[float, ...]
    operating_cost: float
    hot_utility: tuple[float, ...]
    cold_utility: tuple[float, ...]

    def level(self, name: str, period_idx: int = 0) -> UtilityLevelBalance:
        for row in self.balances:
            if row.level == name and row.period_idx == period_idx:
                return row
        raise KeyError((name, period_idx))

    def marginal_prices(self, period_idx: int = 0) -> dict[str, float]:
        """Each level's marginal price in one period."""
        return {
            row.level: row.marginal_price
            for row in self.balances
            if row.period_idx == period_idx
        }


__all__ = [
    "LevelLoad",
    "SiteUtilityLevel",
    "SiteUtilityLevels",
    "UtilityLevelBalance",
    "UtilitySystemBalance",
]
