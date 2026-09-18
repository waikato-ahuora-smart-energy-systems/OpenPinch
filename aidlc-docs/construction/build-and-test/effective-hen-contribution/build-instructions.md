# Build verification

No packaging or dependency changes. Import/collection is exercised by pytest.
Run `.venv/bin/ruff check` on arrays.py and the three changed test modules,
and `git diff --check`. Both checks passed.
