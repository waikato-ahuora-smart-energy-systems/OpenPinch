# Application Problem Internals

## Overview
Private package holding the implementation behind `PinchProblem` (`OpenPinch/application/problem.py`),
so that file stays an orchestrator. Input loading and validation, result shaping, multi-period
summaries, targeting execution and the public accessors each live in a subdirectory. The only module
at this level, `arguments.py`, resolves public workflow arguments against options and configuration.

## Interface
- `ArgumentSpec`, `EffectiveArguments` (`arguments.py`): frozen mapping of a public argument to a configuration key and default; resolved values with provenance labels.
- `MISSING` (`arguments.py`): sentinel for an absent argument or default.
- `LOAD_TIME_OPTION_KEYS` (`arguments.py`): options consumed when the zone tree is built.
- `resolve_effective_arguments` (`arguments.py`): resolve values from `named`, `options`, `config` then `default`, tagged by source.
- `split_runtime_and_configuration_options` (`arguments.py`): separate execution context from flat configuration options.
- `temporary_zone_configuration` (`arguments.py`): context manager applying validated overrides to a zone tree.
- Subdirectories:
  - [accessors](./accessors/README.md): the `target`, `design`, `plot` and `components` accessors on `PinchProblem`.
  - [input](./input/README.md): source loading, validation and zone-tree construction.
  - [output](./output/README.md): result extraction and report shaping.
  - [periods](./periods/README.md): multi-period replay and aggregation.
  - [targeting](./targeting/README.md): method catalogue, execution, transactions and provenance.

## Invariants
- Precedence is named argument, then `options` (by public name, then option key), then configuration value, then default. [arguments.py:resolve_effective_arguments]
- Resolved values and provenance are read-only mappings. [arguments.py:EffectiveArguments]
- An option that selects a descriptive workflow method cannot be passed through `options`. [arguments.py:split_runtime_and_configuration_options]
- Load-time options cannot be overridden per call; use `update_options(...)` or a workspace scenario. [arguments.py:temporary_zone_configuration]
- Zone configurations are restored after the override scope, including on error. [arguments.py:temporary_zone_configuration]

## Side Effects
- `temporary_zone_configuration` replaces `zone.config` on every zone of the tree for the duration of the block. [arguments.py:temporary_zone_configuration]

## Verification
- Run: `uv run --no-sync pytest tests/application/test_effective_arguments_properties.py`
- For the whole package: `uv run --no-sync pytest tests/application`.
