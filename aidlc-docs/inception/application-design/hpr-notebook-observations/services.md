# HPR correction services and lifecycle

## S1: Calculate and publish one HPR target

1. Resolve effective load options, mode, placement, zone and selected period.
2. Compute available and selected service; run the relevant HPR model with
   separate service, ambient and feasibility accounting.
3. Verify cycle and residual consistency, including the full residual thermal
   profile and existing utility allocation. Failure must not publish a
   successful eligible residual target.
4. Construct the load summary, HPRResidualData and matching HP or refrigeration
   graph records from that verified result. Keep backend objects out of values.
5. The application binds provenance and a deterministic digest of the relevant
   result data to form the final HPRResidualSnapshot, then atomically publishes
   target, report and graph snapshots. No partially stamped snapshot is exposed.

Do not derive a second residual after reporting from rounded graph coordinates.
The numerical representation precedes and drives presentation. Different
backends share the same output contract; their equations remain backend-owned.

## S2: Observe an HPR result

Reading duty values and profiles returns immutable numerical records. Plot
selection first resolves the requested mode and target; it then retrieves that
target's cached graph set and renders or returns graph data. Graph transport
preserves name, series identity, vertical/utility flags and target/period
context. Selection errors leave the study unchanged. No observation invokes
targeting, fills a missing result by rerunning it or changes configuration.

## S3: Detach the residual

`problem.target.residual_utility(base_target=...)` validates ownership, freshness, supported
classification, scalar period, physical basis and result digest. The new input
contains residual_basis, copied utility definitions, appropriate thermal and
economic settings, and explicit selected-period/zone metadata. It contains no
original process-stream list, HEN network or process components.

Construct the new PinchProblem from this canonical input and return it unsolved.
The new case has its own analysis owner identity; the snapshot retains the
origin identity as historical provenance. Once constructed, it is independent
of the original object's future state. Repeated conversion of an unchanged
target has equal engineering content, while case-owner identities may differ.

`residual_basis` must survive all canonical input and result-case reconstruction
paths used by utility placement. A residual study cannot degrade into an empty
or original process case when cloned or serialized. A digest validates the
stored snapshot content and its claimed selection; it is an integrity check,
not a security/authentication guarantee.

## S4: Allocate utilities on the residual

Target dispatch checks the input kind before ordinary process preprocessing.
On a residual-only case, `direct_heat_integration()` invokes the residual
allocation service. That service uses the stored full-precision profile,
thermal coordinate basis and copied utility definitions, then emits a distinct
residual utility target and its GCC/net-load plots.

The selected period is the snapshot's canonical period. Omission selects that
period; a conflicting period or subzone selection fails explicitly. Reporting
keeps original plant heat recovery and frozen HPR work separate from residual
utility quantities. It must not present recomputed recovery of nonexistent
physical streams. A fully served residual yields zero utility duties without
inventing a dummy stream or temperature interval.

## S5: Optimize residual utility placement

1. Normalize utility templates, level counts, limits and options through the
   existing utility-placement request boundary.
2. Detect residual_basis and construct the residual placement context directly.
   Skip the process-only `_extract_period()` replay for this case kind.
3. Evaluate every candidate through the frozen residual allocation adapter.
   Reuse the existing optimizer and candidate result contract.
4. For thermal feasibility, use allocation coordinates with their explicit
   shift convention. For entropy/lost-work evaluation, use the separately
   preserved physical boundary with actual temperatures. Candidate utility
   temperatures are converted consistently for each use.
5. Return a detached PinchProblem containing the same residual_basis and the
   selected utility definitions, plus normal placement result evidence. Its
   subsequent allocation reproduces the winning candidate within tolerance.

The existing entropy evaluator currently expects physical composites. It may
need an internal residual-basis branch, but it must not be fed shifted GCC
temperatures under a `real_temperatures` label. The verification boundary is
utility completion conditional on fixed HPR/ambient exchange; any fixed HPR
equipment contribution must be identified separately. This is not a claim of
joint plant optimization or automatically validated whole-plant exergy.

If a target lacks a verified physical basis, reject residual utility-placement
eligibility explicitly rather than fabricating entropy or falling back to
process-only replay. Producing that basis for the supported Carnot notebook
cases is required work, not an optional fallback.

## Lifecycle and error policy

| Event | Behavior |
|---|---|
| Source changes before conversion | Reject a stale source target. |
| Source changes after conversion | Detached residual remains a valid historical study. |
| Utility definitions change in residual case | Invalidate utility results; retain frozen thermal basis. |
| Attempt to resize HPR or alter process shifts in residual case | Explicit unsupported-operation error. |
| Weighted result selected | Reject; require a supported scalar period result. |
| Target/graph record mutated or missing | Reject invalid selection; no hidden regeneration. |
| Candidate infeasible | Use existing explicit candidate diagnostics; do not alter source or frozen basis. |
| JSON input malformed or mixed with process data | Validation error before any engineering execution. |

There are no new services outside the Python process, message queues, workers
or external persistence dependencies in this design.
