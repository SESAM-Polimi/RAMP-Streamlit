# utils.py — helpers + IO for Streamlit RAMP app (refactored)
from __future__ import annotations

import json
from typing import List, Tuple
import importlib
from pathlib import Path
from dataclasses import dataclass
import shutil

import numpy as np
import pandas as pd

from config.path_manager import PM

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CONFIG_DIR = PM.config_dir
INPUTS_DIR = PM.inputs_dir
OUTPUTS_DIR = PM.outputs_dir
ARCH_DIR = PM.archetypes_dir

TEMPLATE_FILE = PM.template_file
FULL_INPUT_XLSX = PM.full_input_xlsx
YEAR_STRUCTURE_JSON = PM.year_structure_json
YEAR_STRUCTURE_YAML = PM.year_structure_yaml
ARCHETYPE_CONFIGS_JSON = PM.archetype_configs_json

# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------
def default_month_partition(k: int) -> List[List[int]]:
    """Partition months 1..12 into k contiguous blocks (reasonable default)."""
    months = list(range(1, 13))
    if k == 1:
        return [months]
    base, extra = divmod(12, k)
    parts, idx = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        parts.append(months[idx: idx + size])
        idx += size
    return parts


def _load_template_df() -> pd.DataFrame:
    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(
            f"RAMP template not found at {TEMPLATE_FILE}. "
            "Place 'ramp_template.xlsx' in /config."
        )
    return pd.read_excel(TEMPLATE_FILE)

def clear_directory(path: Path) -> None:
    """Remove all files and subfolders inside `path` (but keep the folder)."""
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)



# ---------------------------------------------------------------------
# Patch RAMP's np.NaN -> np.nan (idempotent)
# ---------------------------------------------------------------------
def patch_ramp_nan() -> int:
    import ramp
    pkg_path = Path(ramp.__file__).parent
    patched = 0
    for py in pkg_path.rglob("*.py"):
        try:
            txt = py.read_text(encoding="utf-8")
        except Exception:
            continue
        if "np.NaN" in txt:
            py.write_text(txt.replace("np.NaN", "np.nan"), encoding="utf-8")
            patched += 1
    # reload core
    import ramp.core.core as core
    importlib.reload(core)
    return patched


# ---------------------------------------------------------------------
# Time parsing helper
# ---------------------------------------------------------------------
def hhmm_to_minutes(val, is_end: bool = False) -> int:
    """
    Convert a cell to minutes from midnight.

    Rules:
      - 'none', empty, NaN -> 0
      - '00:00' -> 0
      - in END columns, '23:59' variants -> 1440
      - hh:mm -> h*60 + m
      - numeric -> hours (unless very close to 24 in END, then 1440)
    """
    if pd.isna(val):
        return 0

    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "none", "nan"):
            return 0
        if is_end and (
            s.startswith("23:59") or
            s.startswith("23,59") or
            s.replace(".", ":").startswith("23:59")
        ):
            return 1440
        try:
            parts = s.replace(".", ":").split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            return h * 60 + m
        except Exception:
            try:
                hours = float(s.replace(",", "."))
                return int(round(hours * 60))
            except Exception:
                return 0

    if hasattr(val, "hour") and hasattr(val, "minute"):
        if is_end and val.hour == 23 and val.minute == 59:
            return 1440
        return int(val.hour) * 60 + int(val.minute)

    if isinstance(val, (int, float, np.integer, np.floating)):
        if is_end and abs(float(val) - 24.0) < 1e-6:
            return 1440
        return int(round(float(val) * 60))

    return 0

def matrix_to_long_series(
    mat: np.ndarray,
    freq: str,
    year: int = 2019, # non-leap year
    col_name: str = "power_W",
    add_datetime_index: bool = True,
) -> pd.DataFrame:
    """
    Convert a (days, steps_per_day) matrix into a long dataframe with length days*steps_per_day.

    freq:
      - "H"  -> expects steps_per_day = 24
      - "T"  -> expects steps_per_day = 1440 (minute)
      - "15T" etc also possible (then steps_per_day must match)

    Output:
      - if add_datetime_index=True: columns: ["datetime", col_name]
      - else: columns: [col_name] only (single column)
    """
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix (days, steps_per_day), got shape={mat.shape}")

    days, steps_per_day = mat.shape
    values = mat.reshape(days * steps_per_day, order="C")  # day0 all steps, then day1, ...

    if not add_datetime_index:
        return pd.DataFrame({col_name: values})

    # Build a proper timeline for the chosen year
    start = pd.Timestamp(f"{year}-01-01 00:00:00")
    dt_index = pd.date_range(start=start, periods=len(values), freq=freq)

    return pd.DataFrame({"datetime": dt_index, col_name: values})

# ---------------------------------------------------------------------
# Archetype presets for cyclic appliances
# ---------------------------------------------------------------------
freezer_params = dict(
    power=200, num_windows=1, func_time=1440, time_fraction_random_variability=0,
    func_cycle=30, fixed="yes", fixed_cycle=3, occasional_use=1, flat="no",
    thermal_p_var=0, pref_index=0, wd_we_type=2,
    p_11=200, t_11=20, cw11_start=580, cw11_end=1200,
    p_12=5, t_12=10, cw12_start=0, cw12_end=0, r_c1=0,
    p_21=200, t_21=15, cw21_start=510, cw21_end=579,
    p_22=5, t_22=15, cw22_start=0, cw22_end=0, r_c2=0,
    p_31=200, t_31=10, cw31_start=0, cw31_end=509,
    p_32=5, t_32=20, cw32_start=1201, cw32_end=1440, r_c3=0,
    window_1_start=0, window_1_end=1440, window_2_start=0, window_2_end=0,
    window_3_start=0, window_3_end=0, random_var_w=0,
)

fridge_params = dict(
    power=150, num_windows=1, func_time=1440, time_fraction_random_variability=0,
    func_cycle=30, fixed="yes", fixed_cycle=3, occasional_use=1, flat="no",
    thermal_p_var=0, pref_index=0, wd_we_type=2,
    p_11=150, t_11=20, cw11_start=580, cw11_end=1200,
    p_12=5, t_12=10, cw12_start=0, cw12_end=0, r_c1=0,
    p_21=150, t_21=15, cw21_start=420, cw21_end=579,
    p_22=5, t_22=15, cw22_start=0, cw22_end=0, r_c2=0,
    p_31=150, t_31=10, cw31_start=0, cw31_end=419,
    p_32=5, t_32=20, cw32_start=1201, cw32_end=1440, r_c3=0,
    window_1_start=0, window_1_end=1440, window_2_start=0, window_2_end=0,
    window_3_start=0, window_3_end=0, random_var_w=0,
)


# ---------------------------------------------------------------------
# Converters: simplified DF → full DF
# ---------------------------------------------------------------------
def build_full_from_simplified_df(simplified: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, s in simplified.iterrows():
        user_cat   = s["User category"]
        num_users  = s["Number of users"]
        app_name   = s["Appliance name"]
        app_number = s["Appliance number"]

        raw_cycle = s.get("Cycle Archetype", "none")
        cycle = "none" if pd.isna(raw_cycle) else str(raw_cycle).strip().lower()

        # Base row from template
        base = template[(template["name"] == app_name) & (template["user_name"] == user_cat)]
        if base.empty:
            base = template[template["name"] == app_name]
        base_row = (base.iloc[0] if not base.empty else template.iloc[0]).copy()

        # Common fields
        base_row["user_name"] = user_cat
        base_row["num_users"] = num_users
        base_row["name"]      = app_name
        base_row["number"]    = app_number

        if cycle == "freezer_default":
            for k, v in freezer_params.items():
                if k in base_row.index:
                    base_row[k] = v

        elif cycle == "fridge_default":
            for k, v in fridge_params.items():
                if k in base_row.index:
                    base_row[k] = v

        else:
            power       = s["Appliance power [W]"]
            n_fw_raw    = s["Number of functioning windows (FW)"]
            func_time   = s["Total usage duration [mins]"]
            fixed_sched = str(s["Fixed daily schedule [yes/no]"]).strip().lower()
            occ_use     = s["Occasional use [-]"]
            rand_frac   = s["Time fraction of random variability [-]"]

            try:
                n_fw = int(n_fw_raw)
            except Exception:
                n_fw = 0
            n_fw = max(0, min(3, n_fw))  # clamp 0..3

            base_row["power"]        = power
            base_row["num_windows"]  = n_fw
            base_row["func_time"]    = func_time
            base_row["flat"]         = fixed_sched   # 'yes' / 'no'
            base_row["occasional_use"] = occ_use
            base_row["time_fraction_random_variability"] = rand_frac
            base_row["random_var_w"] = rand_frac

            # FW1
            if n_fw >= 1:
                s1 = s["FW 1 - Start (hh:mm)"]
                e1 = s["FW 1 - End (hh:mm)"]
                w1_start = hhmm_to_minutes(s1, is_end=False)
                w1_end   = hhmm_to_minutes(e1, is_end=True)
            else:
                w1_start = 0; w1_end = 0

            # FW2
            if n_fw >= 2:
                s2 = s.get("FW 2 - Start (hh:mm)", "none")
                e2 = s.get("FW 2 - End (hh:mm)", "none")
                w2_start = hhmm_to_minutes(s2, is_end=False)
                w2_end   = hhmm_to_minutes(e2, is_end=True)
            else:
                w2_start = 0; w2_end = 0

            # FW3
            if n_fw >= 3:
                s3 = s.get("FW 3 - Start (hh:mm)", "none")
                e3 = s.get("FW 3 - End (hh:mm)", "none")
                w3_start = hhmm_to_minutes(s3, is_end=False)
                w3_end   = hhmm_to_minutes(e3, is_end=True)
            else:
                w3_start = 0; w3_end = 0

            base_row["window_1_start"] = w1_start
            base_row["window_1_end"]   = w1_end
            base_row["window_2_start"] = w2_start
            base_row["window_2_end"]   = w2_end
            base_row["window_3_start"] = w3_start
            base_row["window_3_end"]   = w3_end

            # Consistency note (non-fatal)
            total_window_time = max(w1_end - w1_start, 0) + \
                                max(w2_end - w2_start, 0) + \
                                max(w3_end - w3_start, 0)
            if total_window_time < func_time:
                print(
                    f"⚠️ windows total ({total_window_time} min) < func_time ({func_time} min) "
                    f"for appliance '{app_name}' | user '{user_cat}'"
                )

        rows.append(base_row)

    return pd.DataFrame(rows, columns=template.columns)


# ---------------------------------------------------------------------
# Streamlit-friendly wrappers: build & save full inputs
# ---------------------------------------------------------------------
def build_full_from_simplified(file_like) -> pd.DataFrame:
    """Reads simplified Excel (file-like), builds full DF, saves to inputs/ramp_input.xlsx."""
    template_df = _load_template_df()
    simplified_df = pd.read_excel(file_like)
    full_df = build_full_from_simplified_df(simplified_df, template_df)
    FULL_INPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_excel(FULL_INPUT_XLSX, index=False)
    return full_df


def build_full_input_for_archetype(file_like, arch_id: str) -> pd.DataFrame:
    """Saves per-archetype full input to inputs/archetypes/ramp_input_<arch_id>.xlsx."""
    template_df = _load_template_df()
    simplified_df = pd.read_excel(file_like)
    full_df = build_full_from_simplified_df(simplified_df, template_df)

    out_path = ARCH_DIR / f"ramp_input_{arch_id}.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_excel(out_path, index=False)
    return full_df


# ---------------------------------------------------------------------
# Year structure + archetype configs IO
# ---------------------------------------------------------------------
def save_year_structure(config: dict, as_yaml: bool = False) -> Path:
    """
    Persist year structure configuration.

    YAML is treated as the authoritative format; JSON is written
    as a mirror for backward compatibility / debugging.
    """
    # Always write JSON mirror
    YEAR_STRUCTURE_JSON.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    if as_yaml:
        import yaml
        YEAR_STRUCTURE_YAML.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        return YEAR_STRUCTURE_YAML

    return YEAR_STRUCTURE_JSON

def load_year_structure() -> dict:
    """
    Load year structure. Prefer YAML (if present), otherwise fall back to JSON.
    """
    if YEAR_STRUCTURE_YAML.exists():
        import yaml
        txt = YEAR_STRUCTURE_YAML.read_text(encoding="utf-8").strip()
        return yaml.safe_load(txt) or {}

    if YEAR_STRUCTURE_JSON.exists():
        return json.loads(YEAR_STRUCTURE_JSON.read_text(encoding="utf-8"))

    return {}



def save_archetype_configs(configs: dict) -> Path:
    ARCHETYPE_CONFIGS_JSON.write_text(json.dumps(configs, indent=2), encoding="utf-8")
    return ARCHETYPE_CONFIGS_JSON


def load_archetype_configs() -> dict:
    if ARCHETYPE_CONFIGS_JSON.exists():
        return json.loads(ARCHETYPE_CONFIGS_JSON.read_text(encoding="utf-8"))
    return {}


def save_dataframe_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------
# Archetype derivation + calendar helpers (single source of truth)
# ---------------------------------------------------------------------
def _safe_id(s: str | None) -> str:
    if s is None:
        return "none"
    return (
        str(s).strip().lower()
        .replace(" ", "_")
        .replace("–", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def derive_archetypes(year_structure: dict) -> list[dict]:
    """
    From a year structure dict -> list of archetype dicts with:
    {season, week_class, label, arch_id}
    """
    n_seasons = int(year_structure.get("n_seasons", 1))
    seasons = year_structure.get("seasons", [])
    use_week = bool(year_structure.get("use_week_classes", False))
    week_classes = year_structure.get("week_classes", [])

    archetypes: list[dict] = []
    if n_seasons == 1 and not use_week:
        return archetypes  # Single archetype handled elsewhere

    if use_week and week_classes:
        for s in seasons:
            s_name = s["name"]
            for wc in week_classes:
                wc_name = wc["name"]
                label = f"{s_name} – {wc_name}"
                arch_id = f"{_safe_id(s_name)}__{_safe_id(wc_name)}"
                archetypes.append({
                    "season": s_name,
                    "week_class": wc_name,
                    "label": label,
                    "arch_id": arch_id,
                })
    else:
        for s in seasons:
            s_name = s["name"]
            label = f"{s_name}"
            arch_id = f"{_safe_id(s_name)}__none"
            archetypes.append({
                "season": s_name,
                "week_class": None,
                "label": label,
                "arch_id": arch_id,
            })
    return archetypes


def build_week_pattern(week_classes: list[dict]) -> list[str] | None:
    """From [{'name','days_per_week'}, ...] -> 7-element list (Mon..Sun)."""
    if not week_classes:
        return None
    patt: list[str] = []
    for wc in week_classes:
        patt.extend([wc["name"]] * int(wc["days_per_week"]))
    if not patt:
        return None
    if len(patt) < 7:
        patt.extend([patt[-1]] * (7 - len(patt)))
    elif len(patt) > 7:
        patt = patt[:7]
    return patt


def month_to_season_map(seasons: list[dict]) -> dict[int, str]:
    """Month(1..12) -> season name; missing months fallback to first season."""
    mapping: dict[int, str] = {}
    if not seasons:
        return {m: "Season 1" for m in range(1, 13)}
    default_season = seasons[0]["name"]
    for s in seasons:
        for m in s.get("months", []):
            mapping.setdefault(int(m), s["name"])
    for m in range(1, 13):
        mapping.setdefault(m, default_season)
    return mapping


def build_calendar_metadata_from_year_structure(n_days: int) -> Tuple[pd.DatetimeIndex | None, list[str] | None, list[str] | None]:
    """
    Use config/year_structure.json to derive (dates, seasons_per_day, weekclasses_per_day).
    Returns (None, None, None) if config missing.
    """
    cfg = load_year_structure()
    if not cfg:
        return None, None, None

    seasons = cfg.get("seasons", [])
    use_week = bool(cfg.get("use_week_classes", False))
    week_classes = cfg.get("week_classes", [])
    m2s = month_to_season_map(seasons)

    dates = pd.date_range("2019-01-01", periods=n_days, freq="D") # based on non-leap year
    seasons_for_days = [m2s.get(d.month) for d in dates]

    if use_week and week_classes:
        patt = build_week_pattern(week_classes)
        if patt is None:
            weekclasses_for_days = [None] * len(dates)
        else:
            weekclasses_for_days = [patt[d.weekday()] for d in dates]
    else:
        weekclasses_for_days = [None] * len(dates)

    return dates, seasons_for_days, weekclasses_for_days


# ---------------------------------------------------------------------
# Results I/O & transforms used in app.py
# ---------------------------------------------------------------------
@dataclass
class ProfileUser:
    name: str
    demand_data: pd.DataFrame  # rows = days, cols = minutes (1440)


def load_profiles_from_outputs() -> tuple[list[ProfileUser], np.ndarray | None]:
    """
    Rebuild users & aggregated profiles from OUTPUTS_DIR CSVs.
    Returns: (users_list, aggregated_matrix_or_None)
    - Includes per-user minute-resolution profiles only (days x 1440).
    - Loads 'profile_aggregated.csv' as aggregated.
    - Skips 'profile_aggregated_hourly.csv' and any non-1440-column files.
    """
    users: list[ProfileUser] = []
    aggregated = None

    if not OUTPUTS_DIR.exists():
        return users, aggregated

    for csv_path in sorted(OUTPUTS_DIR.glob("profile_*.csv")):
        name = csv_path.name

        # Aggregated minute-resolution (ok)
        if name == "profile_aggregated.csv":
            aggregated = pd.read_csv(csv_path).to_numpy()
            continue

        # Explicitly skip hourly aggregate
        if name == "profile_aggregated_hourly.csv":
            continue

        # Load candidate user file
        df = pd.read_csv(csv_path)

        # Only accept minute-resolution (1440 columns)
        if df.shape[1] != 1440:
            # silently skip non-minute profiles
            continue

        # profile_high_income.csv → "high income"
        base = csv_path.stem[len("profile_"):].replace("_", " ")
        users.append(ProfileUser(name=base, demand_data=df))

    return users, aggregated


def load_aggregated_from_outputs() -> np.ndarray | None:
    csv_path = OUTPUTS_DIR / "profile_aggregated.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path).to_numpy()


def build_hourly_from_minute(minute_matrix: np.ndarray) -> np.ndarray:
    """
    Convert minute-resolution aggregated demand (days x 1440) to hourly (days x 24) in W.
    We sum each block of 60 minutes and divide by 60 → average W per hour.
    """
    arr = np.asarray(minute_matrix)
    if arr.ndim == 1:
        if arr.size % 1440 != 0:
            raise ValueError(f"Aggregated length {arr.size} not multiple of 1440.")
        n_days = arr.size // 1440
        arr = arr.reshape(n_days, 1440)
    elif arr.ndim == 2 and arr.shape[1] != 1440:
        raise ValueError(f"Expected 1440 minutes, got {arr.shape[1]}.")
    n_days = arr.shape[0]
    hourly = arr.reshape(n_days, 24, 60).sum(axis=2) / 60.0
    return hourly  # (days, 24)


def save_hourly_aggregated(hourly_matrix: np.ndarray) -> Path:
    df = pd.DataFrame(hourly_matrix, columns=[f"hour_{h:02d}" for h in range(24)])
    out_path = OUTPUTS_DIR / "profile_aggregated_hourly.csv"
    df.to_csv(out_path, index=False)
    return out_path

