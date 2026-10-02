"""Input normalisation helpers shared by Excel/CSV/JSON ingestion paths."""

import warnings

import pandas as pd

__all__ = ["validate_stream_data", "validate_utility_data"]


################################################################################
# Public API
################################################################################


def validate_stream_data(sd: pd.DataFrame):
    """Normalise stream records and fill missing names/zones with sensible defaults."""
    default_zone = "Process Zone"
    default_stream_prefix = "S"

    def _normalize_label(value, prefix: str):
        if value is None or pd.isna(value):
            return value
        original = _label_text(value)
        text = original
        if "." in text:
            text = text.replace(".", "-")
        if text.isdigit():
            text = f"{prefix}{text}"
        if text != original:
            warnings.warn(
                f"Name {original!r} was changed to {text!r}: dots become dashes "
                "and numeric names get a prefix.",
                UserWarning,
                stacklevel=4,
            )
        return text

    def _is_missing(value) -> bool:
        return (
            value is None
            or pd.isna(value)
            or (isinstance(value, str) and not value.strip())
        )

    def _has_content(record: dict) -> bool:
        for value in record.values():
            if not _is_missing(value):
                return True
        return False

    def _normalise_records(records: list[dict]) -> list[dict]:
        cleaned = []
        append = cleaned.append
        previous_zone = None
        used_names = set()
        next_default_name_idx = 1

        for record in records:
            if not isinstance(record, dict):
                continue
            record = record.copy()
            if not _has_content(record):
                continue

            zone = record.get("zone")
            if _is_missing(zone):
                record["zone"] = previous_zone if previous_zone else default_zone
            else:
                zone = zone.strip() if isinstance(zone, str) else zone
                record["zone"] = _normalize_label(zone, "Z")
            previous_zone = record["zone"]

            name = record.get("name")
            if _is_missing(name):
                while f"{default_stream_prefix}{next_default_name_idx}" in used_names:
                    next_default_name_idx += 1
                record["name"] = f"{default_stream_prefix}{next_default_name_idx}"
                next_default_name_idx += 1
                warnings.warn(
                    f"A stream in zone {record['zone']!r} has no name; it was "
                    f"named {record['name']!r}.",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                record["name"] = _normalize_label(name, default_stream_prefix)

            used_names.add(record["name"])
            append(record)

        return cleaned

    if sd is None:
        return []

    if isinstance(sd, pd.DataFrame):
        if sd.empty:
            return sd
        records = sd.replace({pd.NA: None}).to_dict(orient="records")
        return pd.DataFrame(_normalise_records(records))

    return _normalise_records(sd)


def validate_utility_data(ud: pd.DataFrame):
    """Drop utility rows without a name."""
    if ud is None:
        return []

    if isinstance(ud, pd.DataFrame):
        if ud.empty:
            return ud
        if "name" in ud.columns:
            ud = ud.copy()
            ud["name"] = ud["name"].map(
                lambda value: (
                    value if value is None or pd.isna(value) else _label_text(value)
                )
            )
            name_series = ud["name"]
            valid = ~name_series.isna()
            if valid.any():
                valid &= name_series.astype(str).str.strip() != ""
            ud = ud.loc[valid].reset_index(drop=True)
        return ud

    cleaned = []
    append = cleaned.append
    for record in ud:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if name is None or pd.isna(name):
            continue
        if isinstance(name, str) and not name.strip():
            continue
        record = {**record, "name": _label_text(name)}
        append(record)
    return cleaned


def _label_text(value) -> str:
    """Return a name as text; spreadsheets give numeric names as numbers."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()
