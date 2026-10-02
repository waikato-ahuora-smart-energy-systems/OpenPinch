"""Tiny GEKKO-shaped NLP backend on SciPy SLSQP for no-GEKKO model tests.

It supports exactly what the fixed-structure HEN model uses: ``Var``,
``Param``, ``Equation`` (``==`` and ``>=``), ``Intermediate``, ``Minimize``
and arithmetic with ``+ - * / **``. Solved values are written back to
``var.value`` as one-element lists, matching GEKKO.
"""

from __future__ import annotations

import operator
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize


class Expr:
    __hash__ = object.__hash__

    def __init__(self, function: Callable[[np.ndarray], float]) -> None:
        self._function = function

    def __call__(self, x: np.ndarray) -> float:
        return self._function(x)

    def _binary(self, other, op, reverse=False):
        left, right = (_wrap(other), self) if reverse else (self, _wrap(other))
        return Expr(lambda x: op(left(x), right(x)))

    def __add__(self, other):
        return self._binary(other, operator.add)

    def __radd__(self, other):
        return self._binary(other, operator.add, True)

    def __sub__(self, other):
        return self._binary(other, operator.sub)

    def __rsub__(self, other):
        return self._binary(other, operator.sub, True)

    def __mul__(self, other):
        return self._binary(other, operator.mul)

    def __rmul__(self, other):
        return self._binary(other, operator.mul, True)

    def __truediv__(self, other):
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._binary(other, operator.truediv, True)

    def __pow__(self, other):
        return self._binary(other, lambda a, b: np.sign(a) * abs(a) ** b)

    def __neg__(self):
        return Expr(lambda x: -self(x))

    def __eq__(self, other):  # type: ignore[override]
        return Constraint("eq", self - _wrap(other))

    def __ge__(self, other):
        return Constraint("ineq", self - _wrap(other))

    def __le__(self, other):
        return Constraint("ineq", _wrap(other) - self)


class Constant(Expr):
    def __init__(self, value: float) -> None:
        self.value = [float(value)]
        super().__init__(lambda _x: self.value[0])


class Variable(Expr):
    def __init__(self, index: int, value: float, lb, ub) -> None:
        self.index = index
        self.value = [float(value)]
        self.lower = lb
        self.upper = ub
        super().__init__(lambda x: x[self.index])


class Constraint:
    def __init__(self, kind: str, residual: Expr) -> None:
        self.kind = kind
        self.residual = residual


def _wrap(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, Constraint):
        raise TypeError("constraints cannot be used in expressions")
    return Constant(float(value))


class ScipyModel:
    """Collects a GEKKO-style model and solves it with SLSQP."""

    def __init__(self) -> None:
        self.variables: list[Variable] = []
        self.constraints: list[Constraint] = []
        self.objective: Expr | None = None
        self.intermediates: list[Expr] = []
        self.success = False
        self.message = ""

    def Var(self, value=0.0, lb=None, ub=None, name=None, **_kwargs):
        variable = Variable(len(self.variables), value, lb, ub)
        self.variables.append(variable)
        return variable

    def Param(self, value=0.0, name=None, **_kwargs):
        return Constant(value)

    def Intermediate(self, expression, name=None):
        intermediate = Expr(_wrap(expression))
        intermediate.value = [0.0]
        self.intermediates.append(intermediate)
        return intermediate

    def Equation(self, constraint):
        if isinstance(constraint, bool | np.bool_):
            if not constraint:
                raise ValueError("constant constraint is infeasible")
            return constraint
        self.constraints.append(constraint)
        return constraint

    def Minimize(self, expression):
        self.objective = _wrap(expression)

    def solve(self, *, maxiter: int = 1000) -> bool:
        x0 = np.array([v.value[0] for v in self.variables], dtype=float)
        lower = np.array(
            [-np.inf if v.lower is None else v.lower for v in self.variables], float
        )
        upper = np.array(
            [np.inf if v.upper is None else v.upper for v in self.variables], float
        )
        x0 = np.clip(x0, lower, upper)
        # Optimise in scaled coordinates x = scale * z so SLSQP sees O(1) steps.
        span = np.where(np.isfinite(upper - lower), upper - lower, 0.0)
        scale = np.maximum.reduce([np.abs(x0), span, np.ones_like(x0)])

        def unscaled(z):
            return z * scale

        constraints = [
            {"type": c.kind, "fun": (lambda z, c=c: c.residual(unscaled(z)))}
            for c in self.constraints
        ]
        objective_scale = max(1.0, abs(self.objective(x0)))
        result = minimize(
            lambda z: self.objective(unscaled(z)) / objective_scale,
            x0 / scale,
            method="SLSQP",
            bounds=list(zip(lower / scale, upper / scale)),
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        if not _feasible(self.constraints, unscaled(result.x)):
            result = _trust_constr(
                lambda z: self.objective(unscaled(z)) / objective_scale,
                result.x if np.all(np.isfinite(result.x)) else x0 / scale,
                lower / scale,
                upper / scale,
                self.constraints,
                unscaled,
                maxiter,
            )
        x = unscaled(result.x)
        for variable, value in zip(self.variables, x):
            variable.value = [float(value)]
        for intermediate in self.intermediates:
            intermediate.value = [float(intermediate(x))]
        violation = max(
            [
                abs(c.residual(x)) if c.kind == "eq" else max(0.0, -c.residual(x))
                for c in self.constraints
            ]
            or [0.0]
        )
        self.success = violation < 1e-4
        self.message = f"{result.message} (max violation {violation:.2e})"
        self.objective_value = float(self.objective(x))
        return self.success


def _feasible(constraints, x, tol: float = 1e-4) -> bool:
    for constraint in constraints:
        value = constraint.residual(x)
        if constraint.kind == "eq" and abs(value) > tol:
            return False
        if constraint.kind == "ineq" and value < -tol:
            return False
    return True


def _trust_constr(objective, z0, lower, upper, constraints, unscaled, maxiter):
    import warnings

    from scipy.optimize import Bounds, NonlinearConstraint

    equalities = [c for c in constraints if c.kind == "eq"]
    inequalities = [c for c in constraints if c.kind == "ineq"]
    nonlinear = []
    if equalities:
        nonlinear.append(
            NonlinearConstraint(
                lambda z: np.array([c.residual(unscaled(z)) for c in equalities]),
                0.0,
                0.0,
            )
        )
    if inequalities:
        nonlinear.append(
            NonlinearConstraint(
                lambda z: np.array([c.residual(unscaled(z)) for c in inequalities]),
                0.0,
                np.inf,
            )
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return minimize(
            objective,
            np.clip(z0, lower, upper),
            method="trust-constr",
            bounds=Bounds(lower, upper),
            constraints=nonlinear,
            options={"maxiter": 3 * maxiter, "xtol": 1e-10, "gtol": 1e-8},
        )


def scipy_fixed_structure_model():
    """Return a FixedStructureModel subclass solved by :class:`ScipyModel`."""

    from OpenPinch.analysis.heat_exchanger_networks.models.fixed_structure import (
        FixedStructureModel,
    )

    class ScipyFixedStructureModel(FixedStructureModel):
        def setup_model(self) -> None:
            self.m = ScipyModel()
            self.mSuccess = 0
            self.solver_run = None

        def optimise(self, print_output: bool = False) -> None:
            self.mSuccess = 1 if self.m.solve() else 0
            if self.mSuccess:
                self.get_post_process()

    return ScipyFixedStructureModel
