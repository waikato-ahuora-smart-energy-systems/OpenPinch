# Tech Stack Decisions

- Retain Python, NumPy, Pint, and Pydantic.
- Use immutable tuples for public segment views and candidate-copy validation for mutations.
- Use Hypothesis in development dependencies for PBT-02, PBT-03, PBT-07, PBT-08, and PBT-09.
- Add no runtime dependency for this unit.
