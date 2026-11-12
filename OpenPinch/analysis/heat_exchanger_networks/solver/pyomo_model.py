"""Native Pyomo model built through the calls the shared HEN helpers make.

The pinch-design equations and the helpers they share with the GEKKO
stage-wise model create variables, parameters and equations with GEKKO's
method names (``Var``, ``Param``, ``Equation`` and so on). ``PyomoModel``
answers the same calls with Pyomo components, so those helpers build a native
Pyomo model without a string conversion. Expressions are ordinary Pyomo
expressions, so sums of any length build without recursion.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .dependencies import require_synthesis_dependency


def _pyomo() -> Any:
    return require_synthesis_dependency(
        "pyomo.environ",
        package="pyomo",
        purpose="Pyomo heat exchanger network equation model construction",
    )


class PyomoModel:
    """A Pyomo ``ConcreteModel`` with exactly one objective.

    Parameters are fixed variables, so a solve substitutes their values and a
    parameter can be changed between solves. Names are accepted for
    signature compatibility and not used: components live in flat lists.
    ``_path`` is a run directory, created only to hold a solver option file.
    """

    def __init__(self) -> None:
        self._path: str | None = None
        pyo = _pyomo()
        self.model = pyo.ConcreteModel()
        self.model.variables = pyo.VarList()
        self.model.parameters = pyo.VarList()
        self.model.constraints = pyo.ConstraintList()
        self.objective = None

    def Var(
        self,
        value: float | None = 0.0,
        lb: float | None = None,
        ub: float | None = None,
        integer: bool = False,
        name: str | None = None,
    ) -> Any:
        pyo = _pyomo()
        variable = self.model.variables.add()
        variable.domain = pyo.Integers if integer else pyo.Reals
        variable.setlb(lb)
        variable.setub(ub)
        variable.set_value(value, skip_validation=True)
        return variable

    def Param(self, value: float = 0.0, name: str | None = None) -> Any:
        parameter = self.model.parameters.add()
        parameter.fix(value)
        return parameter

    def Equation(self, expression: Any) -> Any:
        return self.model.constraints.add(expression)

    def Equations(self, expressions: Iterable[Any]) -> list[Any]:
        return [self.Equation(expression) for expression in expressions]

    def Intermediate(self, expression: Any, name: str | None = None) -> Any:
        return expression

    def sum(self, values: Iterable[Any]) -> Any:
        return _pyomo().quicksum(values, linear=False)

    def Minimize(self, expression: Any) -> None:
        self._set_objective(expression, sense=_pyomo().minimize)

    def Maximize(self, expression: Any) -> None:
        self._set_objective(expression, sense=_pyomo().maximize)

    def _set_objective(self, expression: Any, *, sense: Any) -> None:
        if self.objective is not None:
            raise ValueError(
                "A Pyomo HEN model has exactly one objective; combine the "
                "terms into one expression instead of adding another."
            )
        self.objective = _pyomo().Objective(expr=expression, sense=sense)
        self.model.objective = self.objective
