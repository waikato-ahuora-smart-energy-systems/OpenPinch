"""Native Pyomo model built through the calls the shared HEN helpers make.

The HEN model equations create variables, parameters and equations through a
small modelling interface (``Var``, ``Param``, ``Equation`` and so on).
``PyomoModel`` answers those calls with Pyomo components, so the equations
build a native Pyomo model. Expressions are ordinary Pyomo expressions, so
sums of any length build without recursion.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .dependencies import require_synthesis_dependency

# Parameters are handed out from dense blocks of fixed variables: allocating
# a block at once is about three times faster than one VarList entry at a
# time. Blocks double in size so small models stay small; unused entries
# appear in no constraint, so solvers never see them.
_FIRST_PARAMETER_BLOCK = 64
_LARGEST_PARAMETER_BLOCK = 8192


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
    signature compatibility and not used: variables live in one list and
    parameters in preallocated blocks.
    ``_path`` is a run directory, created only to hold a solver option file.
    """

    # The Pyomo module, looked up when a model is made. It lives on the
    # class, not the instance, so a model still pickles into a worker.
    _pyo: Any = None

    def __init__(self, *, constraints: bool = True) -> None:
        self._path: str | None = None
        # Without constraints the model only holds values: Equation and the
        # objective are not added, so it can never be solved.
        self.adds_constraints = constraints
        pyo = PyomoModel._pyo = _pyomo()
        self.model = pyo.ConcreteModel()
        self.model.variables = pyo.VarList()
        self._parameter_blocks: list[Any] = []
        self._parameters_left = 0
        self.model.constraints = pyo.ConstraintList()
        self.objective = None
        # Set to start a MIP solve from the variables' current values.
        self.warm_start = False

    def Var(
        self,
        value: float | None = 0.0,
        lb: float | None = None,
        ub: float | None = None,
        integer: bool = False,
        name: str | None = None,
    ) -> Any:
        variable = self.model.variables.add()
        # A VarList entry is real already; setting the domain costs a set
        # lookup, so it is set only for integers.
        if integer:
            variable.domain = self._pyo.Integers
        variable.setlb(lb)
        variable.setub(ub)
        variable.set_value(value, skip_validation=True)
        return variable

    def Param(self, value: float = 0.0, name: str | None = None) -> Any:
        if self._parameters_left == 0:
            self._add_parameter_block()
        parameter = self._free_parameters[-self._parameters_left]
        self._parameters_left -= 1
        # The block is fixed already; a parameter is a real number with no
        # bounds, so Pyomo's domain and bound check has nothing to find.
        parameter.set_value(value, skip_validation=True)
        return parameter

    def _add_parameter_block(self) -> None:
        pyo = self._pyo
        size = min(
            _LARGEST_PARAMETER_BLOCK,
            max(
                _FIRST_PARAMETER_BLOCK,
                sum(len(block) for block in self._parameter_blocks),
            ),
        )
        block = pyo.Var(range(size), dense=True)
        self.model.add_component(f"parameters_{len(self._parameter_blocks)}", block)
        block.fix()
        self._parameter_blocks.append(block)
        self._free_parameters = list(block.values())
        self._parameters_left = size

    def Equation(self, expression: Any) -> Any:
        if not self.adds_constraints:
            return None
        return self.model.constraints.add(expression)

    def Equations(self, expressions: Iterable[Any]) -> list[Any]:
        return [self.Equation(expression) for expression in expressions]

    def Intermediate(self, expression: Any, name: str | None = None) -> Any:
        return expression

    def sum(self, values: Iterable[Any]) -> Any:
        return self._pyo.quicksum(values)

    def Minimize(self, expression: Any) -> None:
        self._set_objective(expression, sense=self._pyo.minimize)

    def Maximize(self, expression: Any) -> None:
        self._set_objective(expression, sense=self._pyo.maximize)

    def is_linear(self) -> bool:
        """True when every active constraint and the objective is linear.

        Parameters are fixed variables, so a parameter times a variable
        counts as linear.
        """

        pyo = _pyomo()
        from pyomo.core.expr import polynomial_degree

        expressions = [
            constraint.body
            for constraint in self.model.component_data_objects(
                pyo.Constraint, active=True
            )
        ]
        if self.objective is not None:
            expressions.append(self.objective.expr)
        return all(
            polynomial_degree(expression) in (0, 1) for expression in expressions
        )

    def _set_objective(self, expression: Any, *, sense: Any) -> None:
        if not self.adds_constraints:
            return
        if self.objective is not None:
            raise ValueError(
                "A Pyomo HEN model has exactly one objective; combine the "
                "terms into one expression instead of adding another."
            )
        self.objective = self._pyo.Objective(expr=expression, sense=sense)
        self.model.objective = self.objective
