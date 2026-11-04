# Value Helpers

## Overview
Private helpers behind `OpenPinch.domain.value.Value`: input-shape predicates and coercion, unit-text normalisation with a bounded grammar check, and scalar and period resolution for value-like data. `value.py` imports `coercion` and `units`; `resolution` is imported directly by the analysis package, and `coercion.coerce_period_index` is shared across the domain and analysis modules.

## Interface
- `ValueCoercion` (`coercion.py`): converts scalars, sequences, mappings, Pint quantities and value-with-unit objects into a Pint quantity and optional weights, handling unit conversion including offset against difference units.
- `frozen_magnitudes` (`coercion.py`): a private read-only 1-D float copy, the form every `Value` stores.
- `coerce_period_index`, `is_bool_like`, `is_numeric_scalar`, `is_array_like_input`, `is_value_with_unit`, `is_serialized_scalar_data`, `is_serialized_period_data` (`coercion.py`): shape predicates and period-index validation.
- `get_scalar_value`, `get_period_value`, `resolve_value_array` (`resolution.py`): resolve one magnitude, or a 1-D array, from scalars, `Value`, serialised mappings and unit-bearing objects.
- `normalise_unit_text`, `check_unit_text`, `UnitTextError`, `MAX_UNIT_TEXT_LENGTH` (`units.py`): unit spelling normalisation and the grammar check.
- `clean_unit_text`, `format_units`, `serialise_units`, `unit_object`, `unit_from_normalised`, `quantity_is_dimensionless`, `same_dimensionality` (`units.py`): display and registry helpers.
- `validate_weights`, `normalise_weights` (`units.py`): period-weight validation, and normalisation to unit sum.

## Invariants
- Unit text is at most 64 characters, products and ratios of names and numbers, with one power of at most two digits per name or group and at most two levels of parentheses. [units.py:check_unit_text]
- Period indexes must be integers, not booleans, and non-negative; strings are converted with `int`. [coercion.py:coerce_period_index]
- Booleans are rejected as numeric values. [coercion.py:ValueCoercion._quantity_from_scalar] [resolution.py:get_period_value]
- A single-element period vector resolves without a `period_idx`; a longer vector defaults to period zero unless `default_allowed` is false, and an out-of-range index raises for vectors with more than one element. [resolution.py:_resolve_period_values]
- Period weights are finite, non-negative and sum to a positive value, and their length must equal the period count. [units.py:validate_weights]
- A dimensionless quantity is converted normally when the requested unit is dimensionless-compatible (fraction to %); only when conversion raises does it adopt the requested unit with its magnitude unchanged. Differing offset and delta temperature units convert as differences. [coercion.py:ValueCoercion._coerce_quantity_to_unit]
- A `Value` never writes its magnitudes in place: stored arrays are read-only, so copies share a quantity (`Value(value)`, `deepcopy`, `to` the same unit), and assignment replaces the quantity. [coercion.py:frozen_magnitudes]
- Empty value sequences are rejected. [coercion.py:ValueCoercion._coerce_magnitude_array]
- Unit spellings: `$` becomes `USD`, `-` and `fraction` become dimensionless, per-year suffixes become `/year`. [units.py:normalise_unit_text]

## Side Effects
- `normalise_unit_text` and `format_units` are memoised with `functools.lru_cache` (256 entries each). [units.py:normalise_unit_text] [units.py:format_units]
- `resolution.py` imports `Value` from the parent `value` module at load. [resolution.py]

## Verification
- Run: `uv run --no-sync pytest tests/domain/test_value.py tests/domain/test_value_resolution.py tests/domain/test_unit_text_grammar.py`
