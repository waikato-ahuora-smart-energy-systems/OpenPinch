# Business Logic Model

## Scope

Unit 2 owns one repository-tool boundary: all OpenHENS code used by a comparison
run must be loaded from the explicitly requested checkout, independent of
ambient `sys.path` order and cached `openhens` modules.

## Exact-Checkout Import Flow

1. Resolve the requested checkout and require it to be a directory.
2. Snapshot the complete `sys.path` sequence and every `sys.modules` entry whose
   name is `openhens` or begins with `openhens.`.
3. Remove those cached entries and place the requested checkout first in the
   temporary import path without duplicate occurrences.
4. Invalidate import caches and import the fixed required module set.
5. Resolve each module's source file and require it to be beneath the requested
   checkout.
6. Validate every required callable without mutating upstream modules.
7. Yield the verified modules for the bounded operation.
8. In `finally`, remove all OpenHENS modules loaded during the scope, restore
   the exact cached-module snapshot and path sequence, and invalidate caches.

## Execution Flow

1. Enter the exact-checkout import scope.
2. Select `OpenHENS` from the verified root module.
3. Pass that callable explicitly to the source-execution helper.
4. Construct and solve the existing model with unchanged arguments.
5. Extract and rank results using the existing comparison logic.
6. Exit the scope and restore interpreter state even when construction, solve,
   or extraction raises.

Preflight enters the same scope without executing a model. Because preflight
occurs before output-directory creation, unsupported checkouts leave no output
tree.

## Transaction Boundary

The context manager is the transaction boundary. Its entry either yields a
fully verified module set or raises after restoration. Its exit always restores
the caller's original OpenHENS cache and import path; it never promises to
preserve unrelated concurrent import mutations because the script is a
single-process command-line comparison utility during this scope.
