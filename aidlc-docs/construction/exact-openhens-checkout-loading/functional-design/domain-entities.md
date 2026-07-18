# Domain Entities

## RequestedCheckout

A resolved filesystem directory supplied through the comparison command. It is
the sole allowed origin for the required OpenHENS modules during the scope.

## RequiredModuleSet

The closed mapping of four module names to imported module objects:

- `openhens`
- `openhens.main`
- `openhens.classes.pinch_classes.process`
- `openhens.classes.pinch_classes.publicOperations`

It is valid only after origin and capability checks pass.

## ImportStateSnapshot

An in-memory snapshot of the complete path list and all pre-existing OpenHENS
module-cache entries. It owns no filesystem or solver state and exists only for
the duration of the import scope.

## VerifiedOpenHENSFactory

The callable `RequiredModuleSet["openhens"].OpenHENS` after all validation. It
is injected into source execution and is not a public OpenPinch contract.

## Relationships

One requested checkout produces at most one valid required module set per
scope. One module set supplies one verified factory. One snapshot surrounds the
entire verification and execution lifetime and is restored exactly once.
