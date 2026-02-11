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

    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
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

