# HPR and MVR Benchmark E2E Requirements Questions

The existing end-to-end targeting suite discovers 54 standard problem files
from `examples/stream_data/p_*.json`. Your clarification establishes that this
problem corpus should feed the HPR/MVR robustness tests. The remaining choices
determine the service matrix and normal-CI cost.

## Question 1
Which existing targeting corpus should feed the HPR/MVR benchmark suite?

A) Reuse the 54 problems discovered by the existing end-to-end targeting suite
(selected from your clarification).

B) Use a named subset of those end-to-end targeting problems.

X) Other (please describe after the [Answer]: tag below).

[Answer]: A

## Question 2
How broad should the HPR targeting matrix be across the standard problems?

A) Run every standard problem through a bounded representative HPR matrix,
covering heat pumping, refrigeration, and optimized MVR while rotating direct,
utility, cascade, and parallel variants deterministically (recommended).

B) Run the full Cartesian product of every standard problem and every HPR
service/topology combination despite the larger CI cost.

X) Other (please describe after the [Answer]: tag below).

[Answer]: A

## Question 3
The standard targeting problems do not define pressure-qualified gas streams
for direct process MVR. How should that service be covered?

A) Add the packaged `process_mvr.json` case as the dedicated direct-MVR
end-to-end benchmark alongside the standard targeting corpus (recommended).

B) Limit this workflow to optimized MVR targeting and leave direct process-MVR
coverage unchanged.

X) Other (please describe after the [Answer]: tag below).

[Answer]: A

## Question 4
How should the real thermodynamic end-to-end matrix run in CI?

A) Keep a deterministic bounded representative matrix in normal CI and reserve
any broader stress matrix for explicit slow/release validation (recommended).

B) Mark every new real CoolProp benchmark test as slow and run it only in
release or manually triggered validation.

X) Other (please describe after the [Answer]: tag below).

[Answer]: A

## Question 5
Should security extension rules be enforced for this test-only workflow?

A) Yes, enforce the Security Baseline as blocking constraints.

B) No, preserve the existing disabled Security Baseline configuration
(recommended).

X) Other (please describe after the [Answer]: tag below).

[Answer]: B

## Question 6
Should the Property-Based Testing extension remain enabled for this workflow?

A) Yes, preserve full Property-Based Testing enforcement (recommended).

B) Partially enforce only pure-function and serialization round-trip rules.

C) No, disable Property-Based Testing for this workflow.

X) Other (please describe after the [Answer]: tag below).

[Answer]: A

## Question 7
Should the resiliency baseline be enabled for this test-only workflow?

A) Yes, apply the Resiliency Baseline as directional design-time guidance.

B) No, preserve the existing disabled Resiliency Baseline configuration
(recommended).

X) Other (please describe after the [Answer]: tag below).

[Answer]: B
