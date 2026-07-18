# Compatibility Shim Representation Questions

The remaining decision concerns equivalent engineering representations rather than retired APIs. Removing these behaviors would change how process engineers provide data, so each is separated below.

## Current Behavior and Consequences

### Unit-group overrides

The flat configuration keys `INPUT_UNIT_TEMPERATURE`, `INPUT_UNIT_PRESSURE`, and similar keys are canonical. A temperature override intentionally applies to both supply and target temperatures; a pressure override applies to both supply and target pressures. The implementation currently calls these shared dimensional groups `aliases`, although they do not preserve retired names.

Retaining the behavior while renaming the internal concept to `unit_groups` or `override_keys` would remove misleading compatibility terminology without making users repeat the same unit configuration for every field.

### Fluid phases

Inputs currently accept the enum member, short engineering codes such as `liq`, descriptive values such as `liquid`, case differences, and both `vapor` and `vapour`. Validation produces one canonical enum-backed value for serialization. This is comparable to accepting equivalent unit spellings, not routing an old API to a new API.

Strict exact-value validation would reject otherwise unambiguous engineering input and would require existing JSON producers to emit only the selected canonical spelling.

### Value inputs

`Value` currently accepts ordinary numeric scalars, one-dimensional period arrays, another `Value`, Pint quantities, the compact serialized mappings used by OpenPinch JSON, Pydantic models with the same data, and foreign objects exposing value and unit attributes.

These forms let process engineers pass calculated quantities without manually unpacking and repacking them. Restricting the forms would simplify validation but would remove current integration functionality rather than remove an old-name shim.

### Contracts that are not interchangeable representations

Compact JSON keys such as `t_supply` are the canonical wire protocol. Optional-dependency guards keep base-package imports lightweight and provide focused installation errors. The segmented-stream parent-axis zero is required by current HEN equation shapes. Removing any of these would be a separate serialization, packaging, or solver redesign.

Please answer each question by adding the selected letter after its `[Answer]:` tag. Choose `X` and add a description when none of the listed options matches your preference.

## Clarification Question 1
How should shared unit overrides be treated?

A) Retain shared dimensional overrides, rename internal `aliases` terminology to `unit_groups` or `override_keys`, and document them as canonical behavior (recommended)

B) Remove shared dimensional overrides and require a separate override for every temperature, pressure, enthalpy, heat-flow, delta-temperature, coefficient, and price field

C) Retain only the flat `INPUT_UNIT_*` and `OUTPUT_UNIT_*` configuration keys and remove direct mapping-based override inputs

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Clarification Question 2
How strict should fluid-phase input be?

A) Retain enum members, short codes, descriptive values, case normalization, and `vapor` or `vapour`, while emitting one canonical serialized value (recommended)

B) Accept only exact canonical descriptive JSON values such as `liquid`, `vapour`, and `gas`

C) Require `FluidPhase` enum members for Python callers and exact canonical descriptive values for JSON callers

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Clarification Question 3
Which `Value` input forms should remain supported?

A) Retain numeric scalars, period arrays, `Value`, Pint quantities, canonical serialized mappings, Pydantic models, and foreign value-with-unit objects as current integration behavior (recommended)

B) Accept only numeric scalars, period arrays, `Value`, and canonical serialized mappings

C) Accept only `Value` objects and canonical serialized mappings at runtime boundaries

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Clarification Question 4
Should compact wire keys, optional-dependency guards, and the segmented-stream solver shape invariant remain unchanged?

A) Yes; classify them as canonical serialization, packaging, and equation contracts rather than compatibility shims (recommended)

B) No; include them in a broader follow-up redesign after this shim cleanup

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A
