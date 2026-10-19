"""Allow-listed solver option names and values for HEN synthesis solvers.

Solver options are written into solver option files and GEKKO option lines,
so only known tolerances, iteration limits and numeric settings are accepted,
each with a number, a boolean or one simple word as its value.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

# IPOPT (through Pyomo or GEKKO).
_IPOPT_OPTIONS = frozenset(
    {
        "tol",
        "acceptable_tol",
        "acceptable_iter",
        "acceptable_obj_change_tol",
        "acceptable_constr_viol_tol",
        "acceptable_dual_inf_tol",
        "acceptable_compl_inf_tol",
        "constr_viol_tol",
        "dual_inf_tol",
        "compl_inf_tol",
        "max_iter",
        "max_cpu_time",
        "max_wall_time",
        "print_level",
        "mu_init",
        "mu_strategy",
        "mu_oracle",
        "bound_push",
        "bound_frac",
        "bound_relax_factor",
        "nlp_scaling_method",
        "nlp_scaling_max_gradient",
        "obj_scaling_factor",
        "hessian_approximation",
        "limited_memory_max_history",
        "linear_solver",
        "max_soc",
        "warm_start_init_point",
        "warm_start_bound_push",
        "warm_start_bound_frac",
        "warm_start_mult_bound_push",
    }
)

# Couenne and its Bonmin options.
_COUENNE_OPTIONS = frozenset(
    {
        "time_limit",
        "node_limit",
        "iteration_limit",
        "solution_limit",
        "allowable_gap",
        "allowable_fraction_gap",
        "integer_tolerance",
        "feas_tolerance",
        "cutoff",
        "cutoff_decr",
        "delete_redundant",
        "log_num_obbt_per_level",
        "max_fbbt_iter",
    }
)

# APOPT.
_APOPT_OPTIONS = frozenset(
    {
        "minlp_maximum_iterations",
        "minlp_max_iter_with_int_sol",
        "minlp_as_nlp",
        "minlp_branch_method",
        "minlp_gap_tol",
        "minlp_integer_tol",
        "minlp_integer_max",
        "minlp_integer_leaves",
        "minlp_print_level",
        "nlp_maximum_iterations",
        "objective_convergence_tolerance",
        "constraint_convergence_tolerance",
        "use_warm_start",
    }
)

ALLOWED_SOLVER_OPTION_NAMES = _IPOPT_OPTIONS | _COUENNE_OPTIONS | _APOPT_OPTIONS

# Integers as large as a solver's 64-bit settings, and short value text.
_MAX_INTEGER = 2**63 - 1
_MAX_TEXT_LENGTH = 32
_NUMBER_TEXT = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d{1,3})?$")
_WORD_TEXT = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")


def validate_solver_option(name: Any, value: Any) -> tuple[str, Any]:
    """Return one allow-listed option, or raise ``ValueError``."""
    if not isinstance(name, str) or name not in ALLOWED_SOLVER_OPTION_NAMES:
        raise ValueError(f"Unsupported solver option {str(name)[:64]!r}.")
    if isinstance(value, bool):
        return name, value
    if isinstance(value, int):
        if abs(value) > _MAX_INTEGER:
            raise ValueError(f"The solver option {name!r} is too large.")
        return name, value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"The solver option {name!r} must be finite.")
        return name, value
    if (
        isinstance(value, str)
        and len(value) <= _MAX_TEXT_LENGTH
        and (_NUMBER_TEXT.fullmatch(value) or _WORD_TEXT.fullmatch(value))
    ):
        return name, value
    raise ValueError(
        f"The solver option {name!r} must be a number, a boolean or a single "
        "word of letters, digits, '_' and '-'."
    )


def validate_solver_options(options: Mapping[Any, Any]) -> dict[str, Any]:
    """Return allow-listed options in insertion order, or raise ``ValueError``."""
    return dict(validate_solver_option(name, value) for name, value in options.items())


__all__ = [
    "ALLOWED_SOLVER_OPTION_NAMES",
    "validate_solver_option",
    "validate_solver_options",
]
