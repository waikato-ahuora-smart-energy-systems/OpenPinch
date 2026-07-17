# Unit 2 Technology Decisions

The existing Python 3.11+ package, Pydantic contracts, NumPy calculations,
pytest, Hypothesis, Ruff, and Sphinx stack is retained. No dependency or
infrastructure change is required. Immutable dataclasses and read-only mapping
views are preferred for replay intent and effective argument provenance.
