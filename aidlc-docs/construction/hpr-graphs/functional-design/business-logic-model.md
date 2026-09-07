# HPR graph design

Resolve the selected local HPR target using application provenance and retained result digests. Render cached numerical graphs only. Default selection requires exactly one eligible target of the requested mode; index selects within that target. Explicit selection also validates zone and mode. No solve or replay occurs.

## Testable Properties

Round trip: graph names and series metadata survive contract serialization. Invariant/idempotence: repeated reads preserve source and result. Oracle: HP and RF selected graphs refer to their own columns. Commutativity/induction N/A. Mutable registry sequences are verified with the residual workflow in unit 3.
