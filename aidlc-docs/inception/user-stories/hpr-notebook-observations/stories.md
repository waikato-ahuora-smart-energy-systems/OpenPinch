# HPR correction user stories

Status: approved by the user's `proceed` response.
Persona: P1, Process Engineer, defined in `personas.md` in this directory.

Requirement references R1 through R7 refer to the numbered approved scope in
`aidlc-docs/inception/requirements/hpr-notebook-observations-requirements.md`.
These are acceptance contracts for the correction, not assertions that the
current implementation passes them. Exact APIs and numerical tolerances remain
functional-design decisions.

## HPR-US-1: Select an HPR service load

As a process engineer, I want to select a fraction or duty of the available
heating or cooling service and see what the cycle actually achieves so that I
can interpret its size without confusing service duty with compressor load.

Requirements: R1, R3. Persona: P1.

### Acceptance scenarios

1. Given a direct background with 750 kW residual heating and 1000 kW residual
   cooling, when I select a fraction of 0.25, then the selected service duty is
   187.5 kW for heating or 250 kW for refrigeration, with that basis identified.
2. Given a utility-system placement or a selected operating period, when I
   select a fraction, then the available duty comes from that placement and
   period, and the result identifies both rather than borrowing another case.
3. Given an economically selected heat pump, when it uses less than the offered
   heating duty, then achieved useful duty is reported separately and remains
   no greater than the selected ceiling within the documented tolerance.
4. Given ambient source or sink exchange, when I inspect sizing, then useful
   process duty, total condenser/evaporator duty, ambient exchange and work are
   distinguishable. Ambient rejection is not counted as useful heating.
5. Given fraction zero, when I run targeting, then no positive HPR service is
   fabricated and no division-by-zero or invalid result occurs. Given fraction
   one, the full available service is offered. Negative, nonfinite and
   above-one fractions receive clear validation errors.
6. Given equivalent valid fraction and explicit-duty selections, when I run
   otherwise identical cases, then selected service and basis agree. Supplying
   conflicting load forms receives a clear error.

## HPR-US-2: Compare economics and operating modes

As a process engineer, I want heat-pump and refrigeration targets to reflect
their service obligations and declared prices so that I can compare useful
opportunities without artificially forcing consumption of source heat.

Requirements: R1, R2, R7. Persona: P1.

### Acceptance scenarios

1. Given a feasible heat-pump candidate with unused low-grade heat, when its
   objective is evaluated, then that unused heat carries its stated residual
   utility cost and no feasibility penalty merely for being unused.
2. Given the selected refrigeration cooling service, when the result reports
   success, then the service obligation is satisfied within the documented
   tolerance and the contributions from the cycle and direct ambient cooling
   are distinguishable. Unserved duty cannot be concealed as cycle service.
3. Given feasible fixed candidates and changed cold/electricity prices, when I
   compare objectives, then only the relevant economic terms change; changing
   prices does not redefine physical feasibility.
4. Given notebook 08's heat-pump study, when I use its explicit small positive
   cooling-price ratio, then the illustrated target can leave source heat for
   external cooling. A controlled sensitivity explains the effect without
   relying on a zero-price ambient-exchange degeneracy or exact solver digits.
5. Given separately configured utility stream prices and HPR price ratios,
   when I inspect the example, then it identifies which settings govern the
   Carnot objective and which govern utility reporting. Global defaults are
   not changed solely to improve the demonstration.
6. Given a mode comparison, when I compare heat pumping with refrigeration,
   then background placement, stage counts and price assumptions are held
   explicit. A separate comparison explains direct versus utility placement.
   Different modes are not required to yield different numbers when the
   physical and economic solution legitimately coincides.

## HPR-US-3: Inspect the selected target's curves

As a process engineer, I want plots to identify and display the target I chose
so that I can distinguish heat-pump and refrigeration results reliably.

Requirements: R4. Persona: P1.

### Acceptance scenarios

1. Given the solved positive-duty heat pump in notebook 08, when I request its
   GCC through the normal public workflow, then I receive finite, nonempty
   traces for that target without searching through empty placeholder graphs.
2. Given a solved refrigeration target, when I request its net-load profiles
   and GCC, then the graphs describe refrigeration rather than returning an
   earlier heat-pump graph. Refrigeration also works without a prior HP solve.
3. Given multiple solved HPR targets, when I select a specific target and
   period, then the returned curves and their visible identification match
   that selection. An unavailable selection does not silently use another.
4. Given graph data passing through result serialization, when it is rendered,
   then graph names, target context and series identification remain available.
5. Given a solved study, when I repeat plot, catalog or graph-data observations,
   then no targeting runs and the engineering state remains unchanged.
6. Given an absent or zero-duty HPR result, when I request an unavailable HPR
   plot, then the outcome explicitly explains its availability rather than
   presenting an unrelated plot as a valid result.

## HPR-US-4: Inspect leftover loads and utilities

As a process engineer, I want the HPR target to provide consistent remaining
thermal loads and utility allocations so that I can reconcile its benefit and
use those loads in the next analysis.

Requirements: R5. Persona: P1.

### Acceptance scenarios

1. Given a solved HPR target, when I inspect its hot and cold utility results,
   then these describe its residual thermal problem without requiring another
   base heat-integration call.
2. Given residual pocket removal introduces temperature breakpoints, when I
   read the after-HPR load profiles, then every profile is finite and aligned
   with its declared temperature grid; the observed all-NaN columns are absent.
3. Given the selected HPR duties and ambient exchange, when I compare cycle
   accounting, residual profiles and allocated utilities, then all describe
   the same physical system and reconcile with independent energy-balance and
   temperature-feasibility checks within documented tolerances.
4. Given physically equivalent near-zero ambient exchanges or negligible
   endpoint rounding, when I compare residual results, then numerical noise
   does not cause a material discontinuity in utility demand or erase profiles.
   Real changes in thermal feasibility remain distinguishable from noise.
5. Given a direct or utility-system target for a selected period, when I read
   its residual, then units, temperature basis, period and ambient contributions
   are explicit and no period's duties are substituted for another's.
6. Given a known zero-HPR case, when residual utilities are evaluated, then the
   original thermal problem is preserved within the documented tolerance.

## HPR-US-5: Optimize utilities on a frozen HPR residual

As a process engineer, I want to allocate or optimize utilities against the
residual of a chosen HPR target so that I can complete its utility study while
keeping the screened HPR arrangement fixed.

Requirements: R6. Persona: P1.

### Acceptance scenarios

1. Given a selected solved HPR target, when I create its residual study through
   the public workflow, then it carries the remaining load, frozen HPR and
   ambient exchange, source/target identity, selected period and temperature
   basis in a detached form.
2. Given this residual study, when I allocate existing utilities, then its
   duties reconcile with the selected HPR residual rather than the original
   process-only load.
3. Given utility-placement optimization on that residual, when any candidate
   is evaluated, then it sees the frozen residual. Changes to utility levels
   do not trigger HPR resizing or a return to the original process-only basis.
4. Given candidate exploration completes, when I inspect the result and the
   source study, then the optimized utilities are attributed to that residual
   and the original source and chosen HPR target remain unchanged.
5. Given a selected supported period, when I repeat the residual workflow, then
   it remains attached to that period. Unsupported aggregate, stale or missing
   selections receive explicit validation rather than an implicit fallback.
6. Given the resulting utility optimum, when I interpret it, then it is clearly
   a sequential result conditional on the frozen HPR target. It is not claimed
   to be a joint optimum over HPR equipment and utility placement.

## HPR-US-6: Reproduce and adapt notebook 08

As a process engineer, I want notebook 08 to execute a complete, understandable
HPR comparison and residual utility study so that I can reproduce the example
and adapt its inputs to my own process.

Requirements: R2, R7. Persona: P1.

### Acceptance scenarios

1. Given a clean notebook kernel and the documented environment, when all cells
   run in order, then both targeting calls complete, both summaries are captured
   after their respective calls, and every displayed plot belongs to its target.
   No undeclared variables or saved execution state are required.
2. Given the duty-selection cells, when I read their explanation and outputs,
   then I can identify the fraction denominator, selected kW, achieved useful
   kW and ambient exchange, without interpreting fraction as compressor load.
3. Given the economic example, when I vary the declared cold/electricity ratio,
   then the outputs explain source use and residual utility changes using a
   controlled case, with stream-price and objective-price roles distinguished.
4. Given a chosen HPR result, when I follow the residual section, then I can
   inspect leftover load profiles, allocate utilities and run the supported
   sequential placement workflow using public methods.
5. Given the documented direct/utility-placement comparison, when I inspect
   results, then the background and stage-count assumptions are explicit.
6. Given the canonical tutorial generator and related guides, when I regenerate
   or cross-reference the example, then the same contract and public workflow
   are documented. Integrating the correction preserves independent user edits.

## Traceability and review

| Requirement | Stories |
|---|---|
| R1: Economic sizing and refrigeration service | HPR-US-1, HPR-US-2 |
| R2: Explicit illustrative economics | HPR-US-2, HPR-US-6 |
| R3: Load-selection contract | HPR-US-1 |
| R4: Correct and identifiable plots | HPR-US-3 |
| R5: Correct finite residuals | HPR-US-4 |
| R6: Detached residual utility workflow | HPR-US-5 |
| R7: Notebook, generator and guides | HPR-US-2, HPR-US-6 |

Each story has one user outcome, a bounded set of observable scenarios and no
prescribed implementation. Stories can be reviewed or tested independently
against prepared inputs. Journey dependencies are explicit: HPR-US-5 consumes
the residual contract in HPR-US-4; HPR-US-6 demonstrates outcomes from the other
five stories. HPR-US-1 and HPR-US-2 share service/accounting definitions, but
verify load interpretation and economic choice separately. This satisfies
INVEST without pretending those real dependencies do not exist.

The full notebook observations supply concrete regression cases. Functional
design must also assess general properties: energy conservation, bounded useful
duty, finite grid-aligned residuals, unit/period consistency, graph transport
round-trips, repeatable observations and non-mutation during residual studies.
These properties complement the concrete examples; generated tests must not
assume a stochastic solver always finds the same optimum.

## Extension compliance at user-stories completion

| Rule | Status | Rationale |
|---|---|---|
| PBT-01 | N/A | Functional design has not started; concerns are carried forward above. |
| PBT-02 | N/A | No transport implementation changed; HPR-US-3 requires preserved graph identity. |
| PBT-03 | N/A | No algorithms changed; numerical invariants are explicit acceptance requirements. |
| PBT-04 | N/A | No idempotent operation implemented; repeated observations are covered by HPR-US-3. |
| PBT-05 | N/A | No algorithm verification performed; independent balance/cascade oracles are required by HPR-US-4. |
| PBT-06 | N/A | No mutable-state implementation changed; HPR-US-5 covers detached candidate sequences. |
| PBT-07 | N/A | No test generators created at this stage. |
| PBT-08 | Compliant | Existing shrinking and reproducibility practices are retained. |
| PBT-09 | Compliant | Existing Hypothesis/pytest stack is reused. |
| PBT-10 | Compliant | Concrete acceptance scenarios and complementary general properties are documented. |
| Security Baseline | N/A | Disabled by existing user configuration. |
| Resiliency Baseline | N/A | Disabled by existing user configuration. |

No blocking extension findings for this stage. Construction compliance remains
pending. Stories and persona are approved for workflow planning.
