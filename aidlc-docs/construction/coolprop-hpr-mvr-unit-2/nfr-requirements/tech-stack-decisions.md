# Unit 2 Technology Stack Decisions

- Pydantic remains the strict public/internal value-contract layer.
- Existing CoolProp fluid resolution remains the property capability owner.
- Existing reusable optimisation models/services receive resolved `maxiter` and
  `maxfun`; they remain HPR-neutral.
- A small private analysis module owns prepared CoolProp facts.
- Python call-local dictionaries provide exact-coordinate caching.
- Pytest and Hypothesis provide examples and fixed-seed properties.
- No dependency, storage, queue, service, or deployment change is required.
