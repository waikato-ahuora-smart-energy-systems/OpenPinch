# Unit dependencies

| Consumer | Unit 1 | Unit 2 | Unit 3 |
|---|---|---|---|
| Unit 1 | Self | None | None |
| Unit 2 | Numerical/identity records | Self | None |
| Unit 3 | Residual and physical basis | Inspectable graphs | Self |
| Unit 4 | Load and cost semantics | Plot API | Residual and placement API |

Execution order 1, 2, 3, 4 is acyclic. All shared files are edited sequentially.
Unit 1 supplies finite exact-grid residual values and actual-temperature thermal
boundary data. Unit 3 retains these through candidate reconstruction. Application
selection/provenance validation is shared by graph selection and conversion.
No backward imports from domain/analysis into application or presentation.
