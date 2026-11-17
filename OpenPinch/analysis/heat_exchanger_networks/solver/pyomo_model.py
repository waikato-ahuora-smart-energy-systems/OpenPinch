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

# Variables and parameters (fixed variables) are handed out from dense
# blocks: allocating a block at once is about three times faster than one
# VarList entry at a time. Blocks double in size so small models stay small;
# unused entries appear in no constraint, so solvers never see them.
_FIRST_BLOCK = 64
_LARGEST_BLOCK = 8192


def _pyomo() -> Any:
    return require_synthesis_dependency(
        "pyomo.environ",
        package="pyomo",
        purpose="Pyomo heat exchanger network equation model construction",
    )


class PyomoModel:
    """A Pyomo ``ConcreteModel`` with exactly one objective.

    Parameters are fixed variables, so a solve substitutes their values and a
    parameter can be changed between solves. Variables and parameters come
    from preallocated blocks; ``variables`` lists the variables handed out,
    in order.
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
        self.variables: list[Any] = []
        self._variable_blocks: list[Any] = []
        self._variables_left = 0
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
    ) -> Any:
        if self._variables_left == 0:
            self._free_variables = self._add_block(self._variable_blocks, "variables")
            self._variables_left = len(self._free_variables)
        variable = self._free_variables[-self._variables_left]
        self._variables_left -= 1
        self.variables.append(variable)
        # A block entry is real already; setting the domain costs a set
        # lookup, so it is set only for integers.
        if integer:
            variable.domain = self._pyo.Integers
        variable.setlb(lb)
        variable.setub(ub)
        variable.set_value(value, skip_validation=True)
        return variable

    def Param(self, value: float = 0.0) -> Any:
        if self._parameters_left == 0:
            self._free_parameters = self._add_block(
                self._parameter_blocks, "parameters"
            )
            self._parameters_left = len(self._free_parameters)
        parameter = self._free_parameters[-self._parameters_left]
        self._parameters_left -= 1
        # The block is fixed already; a parameter is a real number with no
        # bounds, so Pyomo's domain and bound check has nothing to find.
        parameter.set_value(value, skip_validation=True)
        return parameter

    def _add_block(self, blocks: list[Any], kind: str) -> list[Any]:
        """Add the next block of ``kind`` and return its entries.

        Parameter blocks are fixed as a whole when allocated.
        """

        size = min(
            _LARGEST_BLOCK,
            max(_FIRST_BLOCK, sum(len(block) for block in blocks)),
        )
        block = self._pyo.Var(range(size), dense=True)
        self.model.add_component(f"{kind}_{len(blocks)}", block)
        if kind == "parameters":
            block.fix()
        blocks.append(block)
        return list(block.values())

    def Equation(self, expression: Any) -> Any:
        if not self.adds_constraints:
            return None
        return self.model.constraints.add(expression)

    def import_duals(self) -> None:
        """Ask the solver for constraint duals (an imported ``dual``
        suffix); solvers that report none leave them unset."""

        if getattr(self.model, "dual", None) is None:
            self.model.dual = self._pyo.Suffix(direction=self._pyo.Suffix.IMPORT)

    def dual(self, constraint: Any) -> float | None:
        """``constraint``'s dual from the last solve, or ``None`` if duals
        were not imported or the solver reported none."""

        duals = getattr(self.model, "dual", None)
        if duals is None or constraint not in duals:
            return None
        return float(duals[constraint])

    def Equations(self, expressions: Iterable[Any]) -> list[Any]:
        return [self.Equation(expression) for expression in expressions]

    def Intermediate(self, expression: Any) -> Any:
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
