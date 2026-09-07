# Build instructions

Use the existing .venv Python and build dependencies. Run `.venv/bin/python -m build --no-isolation --outdir /tmp/openpinch-api-dist`, then install the wheel into a temporary target with uv and run the public workflow from outside the checkout. Use `UV_CACHE_DIR=/tmp/openpinch-api-uv-cache` where the home cache is sandbox restricted. No package publication is required.
