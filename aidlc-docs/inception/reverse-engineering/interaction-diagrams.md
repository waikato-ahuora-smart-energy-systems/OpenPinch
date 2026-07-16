# Interaction Diagrams

## Scenario Comparison

```mermaid
sequenceDiagram
    participant User
    participant Workspace as PinchWorkspace
    participant Problem as PinchProblem
    participant Services as Analysis services
    participant Views as Workspace views
    User->>Workspace: Load baseline and define variants
    Workspace->>Problem: Materialize a validated case
    Workspace->>Services: Run configured workflow
    Services-->>Problem: Attach targets and graph data
    Problem-->>Workspace: Return solved case state
    Workspace->>Views: Build serializable summaries and diffs
    Views-->>User: Return comparison view or bundle
```

Text alternative: a workspace stores baseline and variant inputs, materializes each as a problem, runs services, and converts solved problems into serializable comparison views or bundles.

## Heat-Exchanger-Network Synthesis

```mermaid
sequenceDiagram
    participant User
    participant Design as Design accessor
    participant Builder as Task builder
    participant Executor as Local synthesis executor
    participant Backend as Optimization backend
    participant Report as Verification and reporting
    User->>Design: Request enhanced synthesis
    Design->>Builder: Build method tasks and settings
    Builder->>Executor: Submit ordered tasks
    Executor->>Backend: Solve stagewise or decomposition model
    Backend-->>Executor: Return candidate or failure metadata
    Executor->>Report: Assemble, verify, and rank candidates
    Report-->>Design: Return synthesis result and selected network
    Design-->>User: Expose result, manifest, and grid diagram
```

Text alternative: the design accessor builds synthesis tasks, executes optional solver backends, then verifies, ranks, and returns candidate networks and reports.

## Heat-Pump or Refrigeration Targeting

```mermaid
sequenceDiagram
    participant User
    participant Target as Target accessor
    participant Base as Direct or indirect integration
    participant HPR as HPR targeting service
    participant Thermo as Thermodynamic unit model
    participant Optimizer
    User->>Target: Request HPR target
    Target->>Base: Ensure compatible base target exists
    Base-->>HPR: Provide problem table and utility context
    HPR->>Optimizer: Evaluate encoded candidate designs
    Optimizer->>Thermo: Solve states and performance
    Thermo-->>Optimizer: Return feasibility, COP, duty, and cost
    Optimizer-->>HPR: Return selected design
    HPR-->>Target: Attach target and graph effects
    Target-->>User: Return typed HPR target
```

Text alternative: HPR targeting first ensures a compatible heat-integration target, then an optimizer evaluates thermodynamic unit models and returns the selected design and its graph effects.

