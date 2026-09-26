# Automatic version bump flow clarification

Requested sequence: normal review; automatic patch-bump PR; merge; main
validation reusing proven review tests; automatic release publishing.

## Question 1: Where does the generated bump PR merge?

A) Into `develop` (recommended for the current develop-to-main flow). Review and
test the existing develop-to-main PR, then generate a version-only PR into
develop. Merge that bump PR, then merge the updated develop-to-main PR into main.
There are two merges; branch protections may require renewed approval after
the bump. Reuse unaffected test evidence rather than repeat the full suite.

B) Into `main` as a new release PR containing the reviewed changes plus the
bump. The original reviewed PR is superseded rather than separately merged;
the new PR must satisfy main's review/check requirements. Approval does not
automatically transfer from the original PR. Do not automatically close it.

X) Other (describe the desired branches and merge sequence).

[Answer]: A
