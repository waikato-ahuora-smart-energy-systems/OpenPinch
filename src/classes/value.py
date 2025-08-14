from pint import UnitRegistry
from ..lib.schema import ValueWithUnit

ureg = UnitRegistry()
Q_ = ureg.Quantity


class Value:
    def __init__(self, data=None, unit: str = None):
        if data is None:
            self._quantity = Q_(0)
        elif isinstance(data, ValueWithUnit):
            self._quantity = Q_(data.value, data.units)
            try:
                self._quantity.to(unit)
            except:
                pass
        else:
            self._quantity = Q_(data, unit) if unit else Q_(data)

    @property
    def value(self):
        return self._quantity.magnitude

    @value.setter
    def value(self, data):
        self._quantity = Q_(data, self.unit)

    @property
    def unit(self):
        return format(self._quantity.units, "~").replace("°","deg").replace(" ","")

    @unit.setter
    def unit(self, unit_str):
        self._quantity = Q_(self.value, unit_str)

    def to(self, new_unit: str) -> "Value":
        return Value(self._quantity.to(new_unit).magnitude, new_unit)

    def __str__(self):
        return f"{self.value} {self.unit}"

    def __repr__(self):
        return f"Value({self.value}, {repr(self.unit)})"

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __round__(self, ndigits=None):
        return round(self.value, ndigits)

    def __eq__(self, other):
        try:
            if isinstance(other, (int, float)):
                return self._quantity.magnitude == other
            return self._quantity == self._to_quantity(other)
        except Exception:
            return False

    def __lt__(self, other):
        return self._quantity < self._to_quantity(other)

    def __le__(self, other):
        return self._quantity <= self._to_quantity(other)

    def __gt__(self, other):
        return self._quantity > self._to_quantity(other)

    def __ge__(self, other):
        return self._quantity >= self._to_quantity(other)

    def __add__(self, other):
        return self._from_quantity(self._quantity + self._to_quantity(other))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._from_quantity(self._quantity - self._to_quantity(other))

    def __rsub__(self, other):
        return self._from_quantity(self._to_quantity(other) - self._quantity)

    def __mul__(self, other):
        return self._from_quantity(self._quantity * self._to_quantity(other))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._from_quantity(self._quantity / self._to_quantity(other))

    def __rtruediv__(self, other):
        return self._from_quantity(self._to_quantity(other) / self._quantity)

    def _to_quantity(self, other):
        if isinstance(other, Value):
            return other._quantity
        return Q_(other)

    def _from_quantity(self, qty):
        return Value(qty.magnitude, format(qty.units, "~"))

    def to_dict(self):
        return {"value": self.value, "unit": self.unit}

    @classmethod
    def from_dict(cls, data):
        return cls(data["value"], data.get("unit"))


# @pd.api.extensions.register_series_accessor("as_value")
# class ValueAccessor:
#     def __init__(self, series):
#         self.series = series

#     def to(self, unit):
#         return self.series.apply(lambda v: v.to(unit))
