# TODO: write docstring 

def linear_interpolation(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """Performs linear interpolation to estimate y at a given x, using two known points (x1, y1) and (x2, y2)."""
    if x1 == x2:
        raise ValueError(
            "Cannot perform interpolation when x1 == x2 (undefined slope)."
        )
    m = (y1 - y2) / (x1 - x2)
    c = y1 - m * x1
    return m * x + c