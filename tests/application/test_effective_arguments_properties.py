"""Properties for public workflow argument precedence."""

from __future__ import annotations

from hypothesis import given, seed, settings

from OpenPinch.application._problem.arguments import (
    MISSING,
    ArgumentSpec,
    resolve_effective_arguments,
)
from tests.strategies.public_arguments import argument_precedence_cases


@seed(20260715)
@settings(max_examples=100, deadline=None)
@given(case=argument_precedence_cases())
def test_effective_argument_precedence_preserves_falsey_and_none(case):
    named = {} if case.named is MISSING else {"engineering_value": case.named}
    options = {} if case.options is MISSING else {"ENGINEERING_VALUE": case.options}
    config = {} if case.config is MISSING else {"ENGINEERING_VALUE": case.config}

    resolved = resolve_effective_arguments(
        {
            "engineering_value": ArgumentSpec(
                "ENGINEERING_VALUE",
                default=case.default,
            )
        },
        named=named,
        options=options,
        config_values=config,
    )

    if case.named is not MISSING:
        expected, source = case.named, "named"
    elif case.options is not MISSING:
        expected, source = case.options, "options"
    elif case.config is not MISSING:
        expected, source = case.config, "config"
    else:
        expected, source = case.default, "default"

    assert resolved.values["engineering_value"] == expected
    assert resolved.provenance["engineering_value"] == source


@seed(20260715)
@settings(max_examples=50, deadline=None)
@given(case=argument_precedence_cases())
def test_effective_argument_resolution_does_not_mutate_inputs(case):
    named = {"engineering_value": case.named}
    options = {"ENGINEERING_VALUE": case.options}
    config = {"ENGINEERING_VALUE": case.config}
    snapshots = (dict(named), dict(options), dict(config))

    resolve_effective_arguments(
        {"engineering_value": ArgumentSpec("ENGINEERING_VALUE", case.default)},
        named=named,
        options=options,
        config_values=config,
    )

    assert (named, options, config) == snapshots
