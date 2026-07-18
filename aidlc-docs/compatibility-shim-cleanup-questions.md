# Compatibility Shim Cleanup Questions

Please answer each question by adding the selected letter after its `[Answer]:` tag. Choose `X` and add a description when none of the listed options matches your preference.

## Question 1
How broadly should the next compatibility cleanup be scoped?

A) Remove compatibility behavior from the shipped `OpenPinch` package only; explicitly isolated developer tooling may retain it

B) Make the entire repository shim-free, including developer scripts, tests, and documentation (recommended)

C) Address only the confirmed runtime spelling alias and document the remaining repository exceptions

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: B

## Question 2
What should replace the string-based `form` selector in `g_ineq_penalty()`?

A) Introduce a canonical `PenaltyForm` enum and accept enum members only (recommended)

B) Split the behavior into explicitly named square and square-root-smoothing functions, with no selector

C) Keep a string selector but accept only exact canonical literals, without case or spacing aliases

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 3
What should happen to the OpenHENS comparison script's runtime monkeypatch?

A) Remove the monkeypatch, require a supported upstream OpenHENS revision, and fail with a clear prerequisite error otherwise (recommended)

B) Delete the comparison script because it is no longer part of the maintained workflow

C) Keep the unshipped monkeypatch as a documented developer-tool exception

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 4
How should the remaining legacy documentation and stale test terminology be handled?

A) Delete the legacy RTD page and generated-index entry, rename the stale test, and add regression guards (recommended)

B) Retitle the page as contributor-internals guidance without legacy wording, rename the stale test, and keep the page indexed

C) Leave documentation and test names unchanged because they do not alter runtime behavior

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 5
Should the previously approved intentional normalization and solver fallbacks remain outside the compatibility-shim cleanup?

A) Yes; retain unit-group overrides, fluid-phase normalization, compact wire keys, Pint and value-like coercion, optional-dependency guards, solver fallbacks, and solver shape invariants (recommended)

B) Retain input and wire normalization, but remove cross-version solver-call fallbacks

C) Enforce one strict representation everywhere, including units, phases, values, and solver integrations

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: X  -  explain further. 
