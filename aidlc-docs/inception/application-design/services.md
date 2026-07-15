# Services

- Input preparation validates schemas, constructs segments in input order, then adds one parent to its zone.
- Direct and indirect targeting expand segments only inside thermodynamic kernels.
- Capital and area targeting sum segment contributions and deduplicate parent counts.
- HPR and MVR unit models call one shared profile-to-parent builder.
- HEN preparation emits parent axes plus segment tensors; model equations use cumulative parent heat coordinates.
- HEN extraction emits one parent exchanger with nested area contributions.
- Network diagrams and controllability consume only parent topology.
