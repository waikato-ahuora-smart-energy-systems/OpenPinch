"""Semantic validation entry points for canonical problem inputs."""

from __future__ import annotations

from typing import Any


def semantic_issues(problem_input, *, context=None, validation_context=None):
    """Return semantic issues after schema-level validation succeeds."""
    from .validation import _semantic_issues_impl

    return _semantic_issues_impl(
        problem_input,
        context=context if context is not None else validation_context,
    )


def segmented_record_issues(record: Any, **context):
    """Return shared semantic issues for one nested segmented record."""
    from .validation import _segmented_record_issues_impl

    return _segmented_record_issues_impl(record, **context)


def validate_problem_semantics(input_data, *, context=None):
    """Validate semantic invariants and raise the established errors."""
    from .validation import _validate_problem_semantics_impl

    return _validate_problem_semantics_impl(input_data, context=context)
