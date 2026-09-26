"""Bounded valid delivery data for reproducible generated tests."""

from hypothesis import strategies as st

hex40 = st.text(alphabet="0123456789abcdef", min_size=40, max_size=40)
hex64 = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)


@st.composite
def manifests(draw):
    version = ".".join(
        map(str, draw(st.tuples(*(st.integers(0, 99) for _ in range(3)))))
    )
    return {
        "schema": 1,
        "policy": "delivery-v1",
        "repository": "example/OpenPinch",
        "source_sha": draw(hex40),
        "source_tree": draw(hex40),
        "run_id": draw(st.integers(1, 10**12)),
        "run_attempt": draw(st.integers(1, 20)),
        "version": version,
        "files": {
            f"openpinch-{version}-py3-none-any.whl": draw(hex64),
            f"openpinch-{version}.tar.gz": draw(hex64),
        },
    }
