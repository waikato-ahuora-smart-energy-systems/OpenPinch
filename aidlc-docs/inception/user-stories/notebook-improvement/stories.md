# Notebook Improvement User Stories

## Story Organization

The 18 stories are grouped by execution profile and bounded to one packaged
notebook each. P-01 is the primary actor and P-02 is the reviewer for every
story. Acceptance scenarios use Given/When/Then and direct traceability.

## Base Profile

### NB-01: First Solve and Core Curves

**Notebook**: `01_first_solve_and_core_curves.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want the first solve to present the core target
summary and a decision-relevant curve, so that I can connect utility targets to
the thermal picture behind them.

#### Scenario: Visible conclusion

**Given** the basic study is validated and targeted, **when** the result section
executes, **then** it presents a labelled summary and at least one core curve
inline. `[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the summary and curve are visible, **when** the interpretation is read,
**then** it identifies the utility target and explains the heat-recovery insight.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base profile, **when** all cells run in order, **then** the public
workflow completes without prompts, manual edits, or stored outputs.
`[FR-03, FR-07, FR-08; NFR-01, NFR-02; AC-03, AC-06, AC-08]`

### NB-02: Focused Direct and Total Site

**Notebook**: `02_focused_direct_and_total_site.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want focused and Total Site results visible
together, so that I can distinguish local recovery from site-wide utility
opportunity.

#### Scenario: Visible conclusion

**Given** direct, indirect, and Total Site targets exist, **when** results are
presented, **then** a comparison table and Total Site profile or utility grand
composite curve appear inline. `[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** both scopes are visible, **when** interpretation is read, **then** it
explains process-level recovery versus site utility coordination.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base profile, **when** cells run in order, **then** focused and site
workflows complete without input or private imports.
`[FR-03, FR-07, FR-08; NFR-01, NFR-02; AC-03, AC-06, AC-08]`

### NB-03: Multi-Segment Streams

**Notebook**: `03_multisegment_streams.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want segmented stream preparation shown in a
table, so that I can verify continuity and duty before trusting its target.

#### Scenario: Visible conclusion

**Given** the variable-heat-capacity stream is prepared, **when** results are
presented, **then** segment temperatures, duties, continuity, and the target are
visible in reviewable tables. `[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the tables are visible, **when** interpretation is read, **then** it
explains why contiguous segments and conserved duty matter.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base profile, **when** cells run in order, **then** the in-memory
study executes deterministically without user-supplied values.
`[FR-03, FR-07, FR-08; NFR-01, NFR-02; AC-03, AC-06, AC-08]`

### NB-04: Workspace Cases and Scenarios

**Notebook**: `04_workspace_cases_and_scenarios.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want baseline and scenario cases compared
inline, so that I can see how changed assumptions affect targets.

#### Scenario: Visible conclusion

**Given** baseline and modified cases are targeted, **when** results are
presented, **then** a labelled case comparison shows targets or metrics inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the comparison is visible, **when** interpretation is read, **then** it
identifies which assumption drives the case difference. `[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base profile, **when** cells run in order, **then** case creation,
batch analysis, and comparison complete without interactive selection.
`[FR-03, FR-07, FR-08; NFR-01, NFR-02; AC-03, AC-06, AC-08]`

### NB-05: Workspace Data and Persistence

**Notebook**: `05_workspace_persistence.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want saved and restored workspace state
compared, so that I can verify persistence retains a reusable study.

#### Scenario: Visible conclusion

**Given** a targeted workspace is saved and restored through a temporary path,
**when** results are presented, **then** original and restored identity,
validation, and summary are compared inline. `[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the comparison is visible, **when** interpretation is read, **then** it
explains which persisted fields establish trust. `[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base profile and temporary directory, **when** cells run, **then**
save and load complete without prompts or repository-local artifacts.
`[FR-03, FR-07, FR-08; NFR-01, NFR-02; AC-03, AC-06, AC-08]`

### NB-06: Multiperiod Heat Integration

**Notebook**: `06_multiperiod_heat_integration.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want period targets and the weighted aggregate
shown together, so that I can identify controlling operating periods.

#### Scenario: Visible conclusion

**Given** all periods are targeted, **when** results are presented, **then** a
period-indexed summary and weighted result appear inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the period comparison is visible, **when** interpretation is read,
**then** it identifies the controlling period and explains weighting.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base profile, **when** cells run in order, **then** replay and
aggregation complete without user intervention.
`[FR-03, FR-07, FR-08; NFR-01, NFR-02; AC-03, AC-06, AC-08]`

### NB-07: Area, Cost, and Exergy

**Notebook**: `07_area_cost_and_exergy.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want area, cost, and exergy evidence presented
together, so that I can evaluate savings against equipment and quality trade-offs.

#### Scenario: Visible conclusion

**Given** area, cost, and exergy analyses exist, **when** results are presented,
**then** a summary and exergy or load-profile visualization appear inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the evidence is visible, **when** interpretation is read, **then** it
explains why minimum energy alone does not select the design.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base plotting profile, **when** cells run, **then** analysis and
presentation complete without prompts or hidden recomputation.
`[FR-03, FR-07, FR-08; NFR-01, NFR-02; AC-03, AC-06, AC-08]`

### NB-12: Cogeneration

**Notebook**: `12_cogeneration.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want cogeneration targets and turbine results
presented inline, so that I can assess process heat and power recovery together.

#### Scenario: Visible conclusion

**Given** cogeneration calculations exist, **when** results are presented,
**then** a labelled heat-and-power table or profile appears inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the evidence is visible, **when** interpretation is read, **then** it
explains utility conditions, recoverable power, and process demand.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base profile, **when** cells run, **then** the public cogeneration
workflow completes without user input. `[FR-03, FR-07, FR-08; NFR-01; AC-03, AC-06, AC-08]`

### NB-13: Multiperiod Cogeneration

**Notebook**: `13_multiperiod_cogeneration.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want period-specific and aggregate
cogeneration performance together, so that I can identify annual power drivers.

#### Scenario: Visible conclusion

**Given** all periods are evaluated, **when** results are presented, **then** a
period table and weighted comparison appear inline. `[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the comparison is visible, **when** interpretation is read, **then** it
identifies the controlling period and limitations of a single design point.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base profile, **when** cells run, **then** multiperiod cogeneration
completes without manual case selection. `[FR-03, FR-07, FR-08; NFR-01; AC-03, AC-06, AC-08]`

### NB-14: Energy Transfer

**Notebook**: `14_energy_transfer.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want source-to-sink transfers shown visually
and in a table, so that I can distinguish large targets from credible links.

#### Scenario: Visible conclusion

**Given** transfers are calculated, **when** results are presented, **then** a
transfer table and profile or diagram appear inline. `[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** transfer evidence is visible, **when** interpretation is read, **then**
it explains source, sink, temperature, magnitude, and boundary credibility.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** the base profile, **when** cells run, **then** transfer analysis and
presentation complete without user input. `[FR-03, FR-07, FR-08; NFR-01; AC-03, AC-06, AC-08]`

## Slow-HPR Profile

### NB-08: Carnot Heat Pump and Refrigeration

**Notebook**: `08_carnot_heat_pump_and_refrigeration.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want cycle effects shown against process load,
so that I can see how lift and placement alter utility demand.

#### Scenario: Visible conclusion

**Given** heat-pump and refrigeration targets exist, **when** results are
presented, **then** cycle metrics and a net-load profile appear inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the evidence is visible, **when** interpretation is read, **then** it
explains lift, placement, performance, and utility savings. `[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** slow-HPR extras, **when** cells run, **then** the notebook completes
without runtime choices or presentation-driven reanalysis.
`[FR-03, FR-07, FR-08; NFR-01, NFR-04; AC-03, AC-06, AC-08]`

### NB-09: Vapour Compression and Brayton

**Notebook**: `09_vapour_compression_and_brayton.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want both cycles compared visibly, so that I
can evaluate alternative assumptions on a consistent process basis.

#### Scenario: Visible conclusion

**Given** both cycle analyses exist, **when** results are presented, **then** a
comparison table or paired cycle views show common measures inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** the comparison is visible, **when** interpretation is read, **then** it
explains fluid, lift, efficiency, and boundary effects. `[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** slow-HPR and Brayton extras, **when** cells run, **then** both workflows
complete without prompts or manual state changes.
`[FR-03, FR-07, FR-08; NFR-01, NFR-04; AC-03, AC-06, AC-08]`

### NB-10: Multiperiod Heat Pumps

**Notebook**: `10_multiperiod_heat_pumps.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want heat-pump performance compared across
periods, so that I can assess robustness under variable loads.

#### Scenario: Visible conclusion

**Given** all periods complete, **when** results are presented, **then** period
metrics and a weighted load or performance comparison appear inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** period evidence is visible, **when** interpretation is read, **then**
it explains capacity, lift, performance, and annual-benefit drivers.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** slow-HPR extras, **when** cells run, **then** all period analyses
complete without user intervention. `[FR-03, FR-07, FR-08; NFR-01, NFR-04; AC-03, AC-06, AC-08]`

### NB-11: Process MVR and Cascade

**Notebook**: `11_process_mvr_and_cascade.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want MVR and cascade results shown against
served process duties, so that I can evaluate staging and lift.

#### Scenario: Visible conclusion

**Given** MVR and cascade analyses exist, **when** results are presented, **then**
stage metrics and an applicable load or cascade view appear inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** stage evidence is visible, **when** interpretation is read, **then** it
explains lift, duties, staging, and process placement. `[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** slow-HPR extras, **when** cells run, **then** workflows complete
without prompts or manual tuning. `[FR-03, FR-07, FR-08; NFR-01, NFR-04; AC-03, AC-06, AC-08]`

## Solver Profile

### NB-15: HEN Synthesis and Selection

**Notebook**: `15_hen_synthesis_and_selection.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want ranked candidates and the selected network
shown together, so that I can compare recovery, utility, area, and cost.

#### Scenario: Visible conclusion

**Given** feasible candidates exist, **when** results are presented, **then** a
ranked table and selected network grid appear inline. `[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** rankings and grid are visible, **when** interpretation is read, **then**
it explains why objective rank alone is insufficient. `[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** synthesis extras and solver, **when** cells run, **then** synthesis,
selection, and presentation complete without runtime input.
`[FR-03, FR-07, FR-08; NFR-01, NFR-04; AC-03, AC-06, AC-08]`

### NB-16: Advanced HEN Methods

**Notebook**: `16_advanced_hen_methods.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want advanced methods compared on the same
basis, so that I can understand how method choice changes designs.

#### Scenario: Visible conclusion

**Given** configured methods complete, **when** results are presented, **then**
status, objective, topology, and network-view evidence are comparable inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** method evidence is visible, **when** interpretation is read, **then** it
explains search method, bounds, and starting-network effects. `[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** synthesis extras and solvers, **when** cells run, **then** every method
uses fixed notebook inputs without intervention.
`[FR-03, FR-07, FR-08; NFR-01, NFR-04; AC-03, AC-06, AC-08]`

### NB-17: Multiperiod HEN Synthesis

**Notebook**: `17_multiperiod_hen_synthesis.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want the shared network and period performance
shown together, so that I can identify the controlling period.

#### Scenario: Visible conclusion

**Given** a shared network exists, **when** results are presented, **then** its
grid and period-specific utility or exchanger results appear inline.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** shared and period evidence is visible, **when** interpretation is read,
**then** it identifies the period controlling area or utility and weighting.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** synthesis extras and solver, **when** cells run, **then** shared design
completes without prompts or between-cell edits.
`[FR-03, FR-07, FR-08; NFR-01, NFR-04; AC-03, AC-06, AC-08]`

## Interactive Profile

### NB-18: Results, Plots, Reports, and Exports

**Notebook**: `18_results_plots_reports_exports.ipynb`  
**Personas**: P-01 primary; P-02 reviewer

As a process-engineer learner, I want cached results shown through tables,
plots, reports, and publication surfaces, so that I can prepare a reviewable
study without rerunning analysis.

#### Scenario: Visible conclusion

**Given** validated cached results, **when** presentation executes, **then** an
inline report table and plot or gallery appear before optional exports.
`[FR-04, FR-09; AC-04, AC-10]`

#### Scenario: Engineering meaning

**Given** views and export results are visible, **when** interpretation is read,
**then** it explains method, scope, units, and case checks before publication.
`[FR-05, FR-06; AC-05]`

#### Scenario: Unattended execution

**Given** interactive extras and a temporary output directory, **when** cells
run, **then** presentation and exports complete without prompts or repository
side effects. `[FR-03, FR-07, FR-08; NFR-01, NFR-02; AC-03, AC-06, AC-08]`

## Cross-Cutting Traceability

| Requirement or criterion | Stories | Coverage |
|---|---|---|
| FR-01, AC-01 | NB-01 through NB-18 | Exactly one story owns each packaged notebook. |
| FR-02, NFR-03, AC-02 | NB-01 through NB-18 | Every story is generator-owned and reproducible. |
| FR-03, AC-03 | NB-01 through NB-18 | Every story retains source-only output state. |
| FR-04, FR-09, AC-04, AC-10 | NB-01 through NB-18 | Every notebook has a subject-specific visible conclusion. |
| FR-05, FR-06, AC-05 | NB-01 through NB-18 | Every notebook has specific interpretation and narrative. |
| FR-07, AC-06 | NB-01 through NB-18 | Every notebook executes without user input. |
| FR-08 | NB-01 through NB-18 | Every story uses public workflows and cached results. |
| FR-10, AC-07, AC-09 | NB-01 through NB-18 | Curriculum, operation coverage, and packaging remain intact. |
| NFR-01, NFR-04, AC-08 | NB-01 through NB-18 | Stories are grouped and verified by profile. |
| NFR-02 | NB-01 through NB-18 | Execution avoids local state, absolute paths, and interaction. |
| NFR-05 | NB-01 through NB-18 | Scenarios provide notebook-specific regression targets. |
| NFR-06, AC-12 | NB-01 through NB-18 | Partial PBT applies later when its rules are applicable. |
| NFR-07 | NB-01 through NB-18 | Security and Resiliency remain disabled. |
| AC-11 | NB-01 through NB-18 | No story requires a new dependency. |

## INVEST Verification

| Stories | Independent | Negotiable | Valuable | Estimable | Small | Testable |
|---|---|---|---|---|---|---|
| NB-01 through NB-18 | One notebook per story. | The qualifying public plot or table remains selectable. | Each creates a visible subject-specific learner outcome. | Generator entry and profile checks bound the work. | One notebook is the delivery boundary. | Every story has observable scenarios. |

## Persona Mapping

- P-01 is the primary actor for NB-01 through NB-18.
- P-02 reviews assumptions, visible evidence, interpretation, and
  reproducibility for NB-01 through NB-18.

## Extension Compliance

- **Security Baseline**: N/A; disabled.
- **Resiliency Baseline**: N/A; disabled.
- **Partial PBT**: N/A for story content. NFR-06 and AC-12 ensure later stages
  evaluate PBT-02, PBT-03, PBT-07, PBT-08, and PBT-09 when applicable.
