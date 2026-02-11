from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from types import MappingProxyType
from dataclasses import field


PRETTY_TO_CANON = {
    # Category-level
    "User category": "user_name",
    "Number of users [#]": "num_users",
    "User preference [-]": "user_preference",

    # Appliance identity & size
    "Appliance name": "name",
    "Appliance units per user [#]": "number",

    # Power & usage model
    "Rated power [W]": "power",
    "Number of functioning windows (FW) [1–3]": "num_windows",
    "Total daily use time [min/day]": "func_time",
    "Randomness in daily use time [-]": "time_fraction_random_variability",
    "Minimum time ON [min]": "func_cycle",

    # Flags / variability
    "Fixed daily schedule? [yes/no]": "fixed",
    "Flat profile within windows? [yes/no]": "flat",
    "Probability of use (per day) [-]": "occasional_use",
    "Randomness in window timing [-]": "random_var_w",

    # Other knobs
    "Thermal power variability [-]": "thermal_p_var",
    "Preference index [-]": "pref_index",
    "Weekday/weekend behaviour type [code]": "wd_we_type",

    # Duty-cycle template selection
    "Cycle template ID [#]": "fixed_cycle",

    # Cycle templates (S1)
    "Cycle S1 – Power level A [W]": "p_11",
    "Cycle S1 – Duration A [min]": "t_11",
    "Cycle S1 – Allowed window A start [min]": "cw11_start",
    "Cycle S1 – Allowed window A end [min]": "cw11_end",
    "Cycle S1 – Power level B [W]": "p_12",
    "Cycle S1 – Duration B [min]": "t_12",
    "Cycle S1 – Allowed window B start [min]": "cw12_start",
    "Cycle S1 – Allowed window B end [min]": "cw12_end",
    "Cycle S1 – Randomness factor [-]": "r_c1",

    # Cycle templates (S2)
    "Cycle S2 – Power level A [W]": "p_21",
    "Cycle S2 – Duration A [min]": "t_21",
    "Cycle S2 – Allowed window A start [min]": "cw21_start",
    "Cycle S2 – Allowed window A end [min]": "cw21_end",
    "Cycle S2 – Power level B [W]": "p_22",
    "Cycle S2 – Duration B [min]": "t_22",
    "Cycle S2 – Allowed window B start [min]": "cw22_start",
    "Cycle S2 – Allowed window B end [min]": "cw22_end",
    "Cycle S2 – Randomness factor [-]": "r_c2",

    # Cycle templates (S3)
    "Cycle S3 – Power level A [W]": "p_31",
    "Cycle S3 – Duration A [min]": "t_31",
    "Cycle S3 – Allowed window A start [min]": "cw31_start",
    "Cycle S3 – Allowed window A end [min]": "cw31_end",
    "Cycle S3 – Power level B [W]": "p_32",
    "Cycle S3 – Duration B [min]": "t_32",
    "Cycle S3 – Allowed window B start [min]": "cw32_start",
    "Cycle S3 – Allowed window B end [min]": "cw32_end",
    "Cycle S3 – Randomness factor [-]": "r_c3",

    # Functioning windows
    "FW1 start [min]": "window_1_start",
    "FW1 end [min]": "window_1_end",
    "FW2 start [min]": "window_2_start",
    "FW2 end [min]": "window_2_end",
    "FW3 start [min]": "window_3_start",
    "FW3 end [min]": "window_3_end",
}

# ---------------------------------------------------------------------
# Canonical column list (backend expectation)
# ---------------------------------------------------------------------
CANONICAL_COLS: Tuple[str, ...] = (
    "user_name", "num_users", "user_preference",
    "name", "number",
    "power", "num_windows", "func_time",
    "time_fraction_random_variability", "func_cycle",
    "fixed", "fixed_cycle", "occasional_use", "flat",
    "thermal_p_var", "pref_index", "wd_we_type",
    "p_11", "t_11", "cw11_start", "cw11_end",
    "p_12", "t_12", "cw12_start", "cw12_end", "r_c1",
    "p_21", "t_21", "cw21_start", "cw21_end",
    "p_22", "t_22", "cw22_start", "cw22_end", "r_c2",
    "p_31", "t_31", "cw31_start", "cw31_end",
    "p_32", "t_32", "cw32_start", "cw32_end", "r_c3",
    "window_1_start", "window_1_end",
    "window_2_start", "window_2_end",
    "window_3_start", "window_3_end",
    "random_var_w",
)

REQUIRED_COLS: Tuple[str, ...] = (
    "user_name", "num_users", "name", "number"
)


# ---------------------------------------------------------------------
# Defaults (used to auto-complete missing optional columns)
# Keep this conservative: it should never break RAMP.
# ---------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "user_preference": 1,

    "power": 0.0,
    "num_windows": 0,
    "func_time": 0,
    "time_fraction_random_variability": 0.0,
    "random_var_w": 0.0,
    "func_cycle": 0,

    "fixed": "no",
    "flat": "no",
    "occasional_use": 1.0,
    "thermal_p_var": 0.0,
    "pref_index": 0,
    "wd_we_type": 2,

    "fixed_cycle": 0,

    # duty-cycle template fields
    **{c: 0.0 for c in CANONICAL_COLS if c.startswith(("p_", "t_", "cw")) or c.startswith("r_c")},

    # functioning windows
    "window_1_start": 0, "window_1_end": 0,
    "window_2_start": 0, "window_2_end": 0,
    "window_3_start": 0, "window_3_end": 0,
}


def _to_yesno(x: Any, default: str = "no") -> str:
    if pd.isna(x):
        return default
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _clamp_minutes(x: Any, default: int = 0) -> int:
    v = _to_int(x, default)
    return int(max(0, min(1440, v)))


@dataclass
class SchemaReport:
    used_pretty_headers: Dict[str, str]  # pretty -> canonical
    ignored_columns: List[str]
    missing_required: List[str]
    filled_defaults: List[str]
    warnings: List[str]


@dataclass(frozen=True)
class RAMPInputSchema:
    # immutable tuples are fine as direct defaults
    canonical_cols: Tuple[str, ...] = CANONICAL_COLS
    required_cols: Tuple[str, ...] = REQUIRED_COLS

    # dict defaults must use default_factory
    # use MappingProxyType to keep them read-only in a frozen dataclass
    pretty_to_canon: Dict[str, str] = field(
        default_factory=lambda: MappingProxyType(dict(PRETTY_TO_CANON))
    )
    defaults: Dict[str, Any] = field(
        default_factory=lambda: MappingProxyType(dict(DEFAULTS))
    )

    def to_canonical(
        self,
        df: pd.DataFrame,
        *,
        strict_unknown_cols: bool = False,
        fill_missing_optional: bool = True,
    ) -> Tuple[pd.DataFrame, "SchemaReport"]:
        """
        Accept only:
          - canonical headers
          - pretty headers (mapped by pretty_to_canon)

        If fill_missing_optional=True, missing canonical columns are added with defaults.
        """

        if df is None or df.empty:
            raise ValueError("Uploaded Excel file is empty.")

        df_in = df.copy()

        # 1) Map pretty headers -> canonical (canonical stays canonical)
        new_cols: List[str] = []
        used_pretty: Dict[str, str] = {}
        ignored: List[str] = []

        for c in df_in.columns:
            if c in self.canonical_cols:
                new_cols.append(c)
            elif c in self.pretty_to_canon:
                canon = self.pretty_to_canon[c]
                used_pretty[str(c)] = canon
                new_cols.append(canon)
            else:
                # keep original for now; we'll drop or raise below
                new_cols.append(str(c))
                ignored.append(str(c))

        df_in.columns = new_cols

        # 2) Handle unknown columns
        if ignored and strict_unknown_cols:
            raise ValueError(
                "Unknown column(s) found (not canonical nor pretty):\n- "
                + "\n- ".join(ignored)
            )

        # If not strict: drop unknown columns
        keep = [c for c in df_in.columns if c in self.canonical_cols]
        df_mapped = df_in.loc[:, keep].copy()

        # 3) Check duplicates (e.g., user provided both pretty + canonical for same field)
        col_index = pd.Index(df_mapped.columns)
        if col_index.duplicated().any():
            dcols = col_index[col_index.duplicated()].tolist()
            raise ValueError(
                "Duplicate columns after mapping (likely provided both canonical and pretty for same field): "
                + ", ".join(map(str, dcols))
            )

        # 4) Fill missing columns
        filled_defaults: List[str] = []
        if fill_missing_optional:
            for c in self.canonical_cols:
                if c not in df_mapped.columns:
                    df_mapped[c] = self.defaults.get(c, np.nan)
                    filled_defaults.append(c)

        # 5) Ensure required columns exist
        missing_required = [c for c in self.required_cols if c not in df_mapped.columns]
        if missing_required:
            raise ValueError("Missing required column(s): " + ", ".join(missing_required))

        # Reorder to canonical
        df_mapped = df_mapped.loc[:, list(self.canonical_cols)]

        # 6) Minimal coercions
        # Identifiers
        df_mapped["user_name"] = df_mapped["user_name"].astype(str).str.strip()
        df_mapped["name"] = df_mapped["name"].astype(str).str.strip()

        # ints
        df_mapped["num_users"] = df_mapped["num_users"].map(lambda x: max(0, _to_int(x, 0)))
        df_mapped["number"] = df_mapped["number"].map(lambda x: max(0, _to_int(x, 0)))
        df_mapped["num_windows"] = df_mapped["num_windows"].map(lambda x: max(0, min(3, _to_int(x, 0))))
        df_mapped["func_time"] = df_mapped["func_time"].map(lambda x: max(0, _to_int(x, 0)))
        df_mapped["func_cycle"] = df_mapped["func_cycle"].map(lambda x: max(0, _to_int(x, 0)))
        df_mapped["fixed_cycle"] = df_mapped["fixed_cycle"].map(lambda x: max(0, _to_int(x, 0)))
        df_mapped["user_preference"] = df_mapped["user_preference"].map(lambda x: max(1, _to_int(x, 1)))
        df_mapped["pref_index"] = df_mapped["pref_index"].map(lambda x: max(0, _to_int(x, 0)))
        df_mapped["wd_we_type"] = df_mapped["wd_we_type"].map(lambda x: max(0, _to_int(x, 2)))

        # floats
        for c in ["power", "time_fraction_random_variability", "random_var_w", "occasional_use", "thermal_p_var"]:
            df_mapped[c] = df_mapped[c].map(lambda x: _to_float(x, float(self.defaults.get(c, 0.0))))

        # yes/no
        df_mapped["fixed"] = df_mapped["fixed"].map(lambda x: _to_yesno(x, "no"))
        df_mapped["flat"] = df_mapped["flat"].map(lambda x: _to_yesno(x, "no"))

        # minutes clamp
        for c in ["window_1_start", "window_1_end", "window_2_start", "window_2_end", "window_3_start", "window_3_end"]:
            df_mapped[c] = df_mapped[c].map(lambda x: _clamp_minutes(x, 0))

        # duty-cycle numeric columns
        duty_cols = [c for c in self.canonical_cols if c.startswith(("p_", "t_", "cw")) or c.startswith("r_c")]
        for c in duty_cols:
            df_mapped[c] = df_mapped[c].map(lambda x: _to_float(x, 0.0))

        # 7) Basic warnings
        warnings: List[str] = []
        if (df_mapped["user_name"] == "").any():
            warnings.append("Some rows have empty user_name (user category).")
        if (df_mapped["name"] == "").any():
            warnings.append("Some rows have empty appliance name (name).")

        report = SchemaReport(
            used_pretty_headers=used_pretty,
            ignored_columns=ignored,
            missing_required=missing_required,
            filled_defaults=filled_defaults,
            warnings=warnings,
        )
        return df_mapped, report