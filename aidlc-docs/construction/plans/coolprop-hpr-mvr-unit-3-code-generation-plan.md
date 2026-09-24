# Unit 3 Code Generation Plan

Status: Approved under completion authorization; Part 2 active.

This plan is the single source of truth for Unit 3 Code Generation.

## Part 1

- [x] Read Unit 3 design, direct-MVR owners, notebooks, and integrated gates.
- [x] Define the seven-step TDD/PBT sequence.
- [x] Approve the sequence under the user's completion authorization.

## Part 2

### Step 1 — RED bounds, context, atomicity, and fallback tests
- [x] Add U3-P1 through U3-P12 examples/properties and confirm missing behavior.

### Step 2 — Compression request validation
- [x] Add finite documented lift/ratio bounds before stage iteration.

### Step 3 — Contextual required-state errors
- [x] Add bounded stream/period/stage/fluid error translation with cause.

### Step 4 — Observable named fallbacks
- [x] Add immutable diagnostics for dry-stage and reduced-profile policies.
- [x] Propagate evidence to stage results without numerical regression.

### Step 5 — Notebook and integrated public proof
- [x] Harden notebooks 09 and 11 exception assertions.
- [x] Run packaged direct-MVR and optimized public CoolProp smoke matrix.

### Step 6 — Verification and refactor
- [x] Run focused/full properties and regressions, Ruff, formatting, compilation, docs, and patch hygiene.

### Step 7 — Summary and Build/Test transition
- [x] Generate traceability summary, mark Unit 3 complete, and enter Build and Test.
