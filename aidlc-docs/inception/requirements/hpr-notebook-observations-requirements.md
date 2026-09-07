# HPR targeting and residual workflow requirements

Status: approved by the user's `approve` response. Depth: standard. Type: numerical correctness,
plotting repair and public workflow clarification across existing owners.

Evidence: `hpr-notebook-observations-findings.md` in this directory.

## Proposed scope

1. Preserve economic heat-pump sizing: unused low-grade process heat must not
   incur a feasibility penalty. Retain penalties or constraints for actual
   thermodynamic infeasibility. Make refrigeration's selected cooling-service
   obligation explicit and distinguish direct ambient cooling from cycle duty.
2. Make tutorial economics explicit and consistent. Use a small positive
   cold/electricity ratio for the heat-pump example, show price sensitivity and
   explain the separate stream-price and HPR-price settings. Do not change
   global economic defaults merely to obtain a better tutorial illustration.
3. Define `load_fraction` as the fraction of available selected process or
   utility background duty offered to HPR, with heating/cooling basis clearly
   identified. Recommend a range of zero through one, retain an explicit duty
   alternative, and report available, selected and achieved useful duties.
   Treat the selected heat-pump duty as a ceiling, allowing economic partial
   use. Account for ambient exchange separately so it cannot silently inflate
   reported useful process service. This is a proposed clarification of the
   contract, including validation changes, not a claim about current behavior.
4. Return nonempty, correctly attributed HPR plots. Remove unsolved placeholder
   graphs; add refrigeration plots; provide explicit selection when multiple
   HPR targets exist. Preserve graph/series identification through transport.
   Plotting must remain read-only and must never rerun targeting.
5. Correct residual curve construction and utility accounting, including
   temperature-grid changes, finite stored residual profiles, near-zero
   boundaries and ambient source/sink treatment. Target summaries, cycle
   accounting and utility allocations must describe the same physical case.
6. Expose a clear, detached residual workflow from a selected solved HPR target
   for existing utility allocation and utility-placement optimization. Freeze
   the selected HPR result, preserve source/target/period/temperature-basis
   provenance and prevent candidate evaluation from reverting to the original
   process-only load. Decide the smallest public API during design. Sequential
   residual utility optimization is in scope; joint HPR/utility optimization
   is excluded from this correction.
7. Revise notebook 08 and its generator plus relevant API/fundamentals guides.
   Compare modes on controlled inputs before separately illustrating direct
   versus utility-system placement. Capture both summaries after their calls,
   display each target's own plots and show the residual workflow. Preserve
   independent user notebook edits while integrating the changes.

## Acceptance and testing

- Reproduce empty default GCC and stale refrigeration plot before fixing them;
  verify actual finite traces and target identity afterward.
- Use a fixed accounting oracle to show zero leftover-cold penalty for heat
  pumping. Test price changes independently from feasibility penalties.
- Verify heating and cooling fractions against the selected background and
  units; cover zero, one, invalid inputs, partial use and ambient exchange.
- Use full public sample regressions for residual duties, including the
  pocket-breakpoint NaN case. Check physical energy balance, thermal feasibility
  and agreement between objective accounting and residual utility evaluation.
- Prove that residual utility-placement candidates consume the selected HPR
  residual and leave the original problem and HPR result unchanged.
- Retain period-aware behavior and reject unsupported selections explicitly.
- Execute the revised notebook from a clean kernel and run affected tests,
  documentation/generator checks, lint and appropriate broader numerical gates.

## Extension compliance for requirements analysis

| Rule | Status | Rationale |
|---|---|---|
| PBT-01 | N/A at this stage | Functional design will identify properties for duty accounting, residual transforms and selection. |
| PBT-02 | N/A at this stage | Implementation will cover changed graph/result transport round-trips. |
| PBT-03 | N/A at this stage | Numerical invariants are required above; property tests belong to construction. |
| PBT-04 | N/A at this stage | Design will assess repeat observation and detached conversion idempotence. |
| PBT-05 | N/A at this stage | Independent energy/cascade oracles are required during construction. |
| PBT-06 | N/A at this stage | Design will evaluate HPR-to-utility operation sequences and non-mutation. |
| PBT-07 | N/A at this stage | Reuse constrained domain strategies when implementing properties. |
| PBT-08 | Compliant | Existing Hypothesis shrinking and fixed-seed practices are retained. |
| PBT-09 | Compliant | Existing Python/Hypothesis/pytest stack is reused. |
| PBT-10 | Compliant | Acceptance requires concrete notebook regressions plus broader properties. |
| Security Baseline | N/A | Disabled by existing user configuration. |
| Resiliency Baseline | N/A | Disabled by existing user configuration. |

No blocking extension findings at requirements analysis. Later design and
construction compliance remains to be demonstrated.

## Approval

The user approved these requirements with `approve`. The next stage is focused
user stories, followed by workflow planning, because residual chaining changes
the public engineering workflow. Implementation has not started.
