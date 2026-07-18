# Business Rules

- Recovery links are Process to Process and require a positive stage.
- Hot-utility links are Utility to Process; cold-utility links are Process to Utility.
- Unassigned and former lowercase endpoint classifications are invalid.
- Period indices are contiguous and zero-based; IDs are unique and aligned
  across every exchanger in a network.
- JSON-visible values are finite and obey existing positivity, range, and area rules.
- Private runtime metadata is not part of the transport contract.
- A canonical runtime JSON dump is reproduced exactly after TargetInput validation.
