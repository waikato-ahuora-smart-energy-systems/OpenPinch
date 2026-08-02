# Notebook Improvement Personas

## Persona Selection

The approved plan defines two personas. Both care about the observable notebook
experience; neither represents a separate implementation unit.

## P-01: Mixed-Experience Process-Engineer Learner

### Profile

A technically capable process engineer who understands process data and energy
integration concepts but may be new to OpenPinch, its public Python workflow, or
some advanced analysis methods.

### Goals

- Run a complete study without supplying input during execution.
- See a useful engineering result rather than a sequence of assignments.
- Understand what the result means and which decision it informs.
- Adapt the demonstrated workflow to plant data with confidence.

### Behaviors and Needs

- Executes notebooks from top to bottom and inspects inline results.
- Uses sample data to learn before substituting site data.
- Needs clear prerequisites, assumptions, ordered steps, and public API examples.
- Needs profile-appropriate plots, tables, reports, or network views followed by
  subject-specific interpretation.

### Pain Points and Constraints

- Cells calculate useful objects but never present them.
- Generic interpretation does not explain an engineering decision.
- Tutorials require edits, prompts, local paths, or unstated dependencies.
- Python proficiency varies; optional profiles require declared extras.

### Success Signals

- Every tutorial runs unattended under its declared profile.
- The concluding inline result is immediately reviewable.
- The learner can state what to inspect and how to adapt the example.

## P-02: Technical Reviewer or Trainer

### Profile

A senior process engineer, technical lead, educator, or internal trainer who
reviews tutorial correctness and uses the series to explain OpenPinch workflows.

### Goals

- Confirm that each tutorial demonstrates a complete, credible study flow.
- Use inline results to discuss trade-offs without extra presentation material.
- Verify that interpretation is specific and supported by visible evidence.
- Trust reproducible execution across supported profiles.

### Behaviors and Needs

- Reviews assumptions and outputs before endorsing a workflow.
- Compares related notebooks across the curriculum.
- Needs consistent structure, visible decision evidence, and clear runtime and
  profile expectations.
- Uses notebooks in guided study, design review, or training sessions.

### Pain Points and Constraints

- Tutorials end before presenting the decision-relevant result.
- Decorative plots do not support the learning outcome.
- Narrative depth varies between profiles.
- Packaged notebooks must remain source-only and generator-owned; external
  solver demonstrations depend on declared prerequisites.

### Success Signals

- Each tutorial supports a clear conversation from assumptions to result.
- The visible conclusion and interpretation match the tutorial subject.
- Execution and source-only checks provide reproducible evidence.

## Persona-to-Story Map

| Persona | Stories | Relationship |
|---|---|---|
| P-01 Mixed-Experience Process-Engineer Learner | NB-01 through NB-18 | Primary actor executing, inspecting, interpreting, and adapting every tutorial. |
| P-02 Technical Reviewer or Trainer | NB-01 through NB-18 | Reviewer of assumptions, visible evidence, interpretation, and decision quality. |

## Extension Compliance

- **Security Baseline**: N/A; disabled.
- **Resiliency Baseline**: N/A; disabled.
- **Partial PBT**: N/A; persona definitions create no pure transformation or
  serialization round trip.
