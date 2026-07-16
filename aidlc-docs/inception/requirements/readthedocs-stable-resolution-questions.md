# Read the Docs Stable-Version Resolution

## Confirmed Root Cause

Read the Docs maps `stable` to the greatest final semantic version exposed by
the remote Git repository. The remote currently exposes only `v0.0.1`, which
points to commit `5c1e44ce613a5decb0b2440dd3ede3340ced06d5`. That commit predates
`.readthedocs.yaml` and the Sphinx documentation tree.

OpenPinch 0.4.5 is the current published package. Its release commit,
`90ae88ff8d627279b6479fafd0d97ffe722cdbd9`, contains the valid Read the Docs
configuration, but no corresponding release tag is present on the remote.

## Question 1
Which version policy should resolve the hosted documentation failure?

A) Add an immutable `0.4.5` tag at release commit `90ae88ff` so Read the Docs
maps `stable` to the published OpenPinch 0.4.5 documentation without
retriggering the `v*.*.*` package-publishing workflow (recommended)

B) Add and maintain a literal `stable` branch at release commit `90ae88ff`,
advancing it manually with each supported release

C) Use `latest` from the default branch as the public documentation, set it as
the Read the Docs default, and deactivate the obsolete `stable` version

D) Leave `stable` unresolved until the next formal `vX.Y.Z` release tag is
created and retained; use `latest` in the meantime

X) Other (please describe after the `[Answer]:` tag below)

[Answer]:
