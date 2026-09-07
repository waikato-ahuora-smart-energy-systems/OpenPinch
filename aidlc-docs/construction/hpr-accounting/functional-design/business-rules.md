# Accounting rules

Fraction must be finite and in [0, 1]. Zero requested duty retains the existing
no-target return. Duty and per-period requests remain capped by availability.
Ambient duty is reported separately from cycle useful duty. The condenser base
for HP is at most the selected heating load; refrigeration may reject heat to
ambient above process heating demand. Global price defaults are unchanged.
Residual values use kW and Celsius with an explicit real/shifted coordinate flag.
Empty/nonfinite residuals cannot be detached. Use existing numerical tolerance
for zero suppression; never round residual profiles to presentation precision.

Domain, business logic, data flow and scenarios follow approved contracts.
Integration is internal; no frontend/external service. Invalid values raise
ValueError. No further product choice is needed under completion authorization.
