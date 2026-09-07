# Repository-wide audit and repair

Review the current working tree across public application APIs, domain/units,
input/output adapters, numerical analysis, optimisation, plotting, contracts,
packaging, CI and documentation. Use risk-focused code inspection plus concrete
adversarial reproductions; passing tests alone do not establish a clean audit.
Preserve independent changes and saved notebook evidence. No commit or deployment.

- [x] Pass 1: inspect repository integrity and cross-layer public boundaries;
  reproduce, repair and verify confirmed defects.
- [x] Pass 2: re-audit fixes and examine domain/numerical/optimisation boundaries
  and their application interactions; repair and verify findings.
- [x] Pass 3: review resulting interactions and repository contracts; repair any
  remaining findings and run final appropriate integrated gates.
- [x] Record reviewed areas, findings, tests and the stopping condition. Stop
  early only after a complete pass finds no issues; otherwise finish three passes.

Baseline from the immediately preceding turn: 3160 tests passed, one deliberately
skipped nine-stream solver benchmark; all optional tutorial profiles executed;
combined coverage 96.1459%. Reuse this baseline for unchanged paths and collect
fresh appropriate verification for repairs. PBT enabled; other extensions remain
disabled. Existing implementation authorization covers audit repairs.
