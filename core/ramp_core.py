# core.py — orchestration around RAMP (refactored)
from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy.random import default_rng
from pathlib import Path

from ramp.core.core import UseCase  # RAMP

from core.utils import matrix_to_long_series
from config.path_manager import PM
from core.utils import (
    build_full_input_for_archetype,
    save_archetype_configs,
    load_year_structure,
    load_archetype_configs,
    save_dataframe_csv,
    month_to_season_map,
    build_week_pattern,
)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*'T' is deprecated.*")


@dataclass
class User:
    name: str
    demand_data: pd.DataFrame

# ---------------------------------------------------------------------
# Public API used by app.py
# ---------------------------------------------------------------------
def build_multi_archetypes_from_uploads(rows: Dict[str, dict]) -> Dict[str, dict]:
    """
    rows = {
      arch_id: {
        "season": str, "week_class": Optional[str], "label": str,
        "num_days": int,
        "file_like": BytesIO  # optional; if missing we skip file build but keep metadata
      }, ...
    }

    The function is **authoritative**: the resulting archetype_configs.json
    will contain ONLY the archetypes passed in `rows`.
    """
    built_configs: Dict[str, dict] = {}

    for arch_id, meta in rows.items():
        buf = meta.get("file_like")

        if buf is None:
            # Not uploaded -> skip file build but keep metadata
            built_configs[arch_id] = {
                "season": meta.get("season"),
                "week_class": meta.get("week_class"),
                "label": meta.get("label"),
                "num_days": int(meta.get("num_days", 0)),
            }
            continue

        _ = build_full_input_for_archetype(buf, arch_id)
        full_path = (PM.archetypes_dir / f"ramp_input_{arch_id}.xlsx").resolve()

        built_configs[arch_id] = {
            "season": meta.get("season"),
            "week_class": meta.get("week_class"),
            "label": meta.get("label"),
            "num_days": int(meta.get("num_days", 0)),
            "full_excel_path": str(full_path),
        }

    if not built_configs:
        raise RuntimeError("No archetype configuration to save. Did you define any archetype?")

    # This overwrites the previous file: last configuration is always the canonical one.
    save_archetype_configs(built_configs)
    return built_configs


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _safe_name(s: Optional[str]) -> str:
    if s is None:
        return "none"
    return (
        str(s).lower()
        .replace(" ", "_")
        .replace("–", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

def _fit_to_year(mat: np.ndarray, target_days: int = 365, seed: int = 42) -> np.ndarray:
    """
    Ensure a (days x 1440) matrix becomes exactly `target_days` rows.
    - If days < target: tile whole cycles and append the first `rem` rows.
    - If days > target: downsample without replacement.
    - If equal: return as-is.
    """
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[1] != 1440:
        raise ValueError(f"Expected (days, 1440) minute matrix, got {mat.shape}")

    days = mat.shape[0]
    if days == target_days:
        return mat

    if days < target_days:
        reps = target_days // days
        rem = target_days % days
        if reps > 0:
            tiled = np.vstack([mat] * reps)
        else:
            tiled = np.empty((0, mat.shape[1]))
        if rem > 0:
            pad = mat[:rem, :]
            return np.vstack([tiled, pad])
        return tiled

    # days > target_days → downsample without replacement
    rng = default_rng(seed)
    idx = rng.choice(days, size=target_days, replace=False)
    return mat[idx, :]

# ---------------------------------------------------------------------
# SINGLE-ARCHETYPE
# ---------------------------------------------------------------------
def run_single_archetype(excel_path: Path, num_days: int) -> Dict:
    """
    Use ONE full RAMP input file containing all user categories.
    Generates per-category arrays of shape (num_days, 1440), fits each to 365 days,
    aggregates, and saves CSVs in outputs/.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df_full = pd.read_excel(excel_path, sheet_name=0)
    categories = [str(c) for c in df_full.iloc[:, 0].unique()]

    per_cat_arrays: Dict[str, np.ndarray] = {}
    for cat in categories:
        cat_df = df_full[df_full.iloc[:, 0] == cat]
        buf = BytesIO()
        cat_df.to_excel(buf, index=False)
        buf.seek(0)

        uc = UseCase(name=cat)
        uc.initialize(num_days=num_days, force=True)
        uc.load(buf)
        daily_profiles = uc.generate_daily_load_profiles(flat=False)
        if daily_profiles is None:
            continue

        arr = np.array(daily_profiles)
        if arr.ndim == 1:
            if arr.size % 1440 != 0:
                continue
            arr = arr.reshape((-1, 1440))

        per_cat_arrays[cat] = arr

    if not per_cat_arrays:
        raise RuntimeError("No profiles generated. Check the input Excel format.")

    # 🔁 Fit each category matrix to a full year (365×1440) *before* aggregating
    per_cat_year: Dict[str, np.ndarray] = {cat: _fit_to_year(mat, 365) for cat, mat in per_cat_arrays.items()}
    aggregated = np.sum(list(per_cat_year.values()), axis=0)  # still (365, 1440)

    # Save results
    PM.outputs_dir.mkdir(parents=True, exist_ok=True)

    EXPORT_MINUTE_LONG = True  # export copies for download
    per_user_csv: Dict[str, str] = {}
    per_user_csv_long: Dict[str, str] = {}

    # --- per-category
    for cat, year_mat in per_cat_year.items():
        # 1) Always save WIDE (365 x 1440) for the app
        wide_path = PM.outputs_dir / f"profile_{_safe_name(cat)}.csv"
        save_dataframe_csv(pd.DataFrame(year_mat), wide_path)
        per_user_csv[cat] = str(wide_path.resolve())

        # 2) Optionally save LONG (525600 x 2) for export
        if EXPORT_MINUTE_LONG:
            df_long = matrix_to_long_series(
                year_mat, freq="T", year=2019, col_name="power_W", add_datetime_index=False
            )
            long_path = PM.outputs_dir / f"profile_{_safe_name(cat)}_minute_long.csv"
            save_dataframe_csv(df_long, long_path)
            per_user_csv_long[cat] = str(long_path.resolve())

    # --- aggregated (365 x 1440)
    agg_wide_path = PM.outputs_dir / "profile_aggregated.csv"
    save_dataframe_csv(pd.DataFrame(aggregated), agg_wide_path)

    agg_long_path = None
    if EXPORT_MINUTE_LONG:
        agg_long = matrix_to_long_series(
            aggregated, freq="T", year=2019, col_name="power_W", add_datetime_index=False
        )
        agg_long_path = PM.outputs_dir / "profile_aggregated_minute_long.csv"
        save_dataframe_csv(agg_long, agg_long_path)

    return {
        "mode": "single",
        "num_days": 365,
        "minutes": 1440,
        "categories": categories,
        # Keep this pointing to WIDE so app logic remains consistent
        "aggregated_csv": str(agg_wide_path.resolve()),
        "per_user_csv": per_user_csv,
        # Optional extra pointers (nice-to-have)
        "aggregated_csv_long": str(agg_long_path.resolve()) if agg_long_path else None,
        "per_user_csv_long": per_user_csv_long,
    }


# ---------------------------------------------------------------------
# MULTI-ARCHETYPE
# ---------------------------------------------------------------------
def run_multi_archetype() -> Dict:
    """
    Uses:
      - year_structure.json (seasons, week_classes)
      - archetype_configs.json ({arch_id:{label, num_days, full_excel_path, ...}})

    Steps:
      1) For each archetype & user category, generate pools of daily profiles (num_days x 1440)
      2) Build a 365-day calendar (2019) using season-by-month + optional week pattern
      3) Sample 1 day from the relevant archetype pool for each day, per category
      4) Save per-user (365 x 1440) and aggregated CSVs
    """
    ys = load_year_structure()
    ac = load_archetype_configs()

    if not ys:
        raise RuntimeError("Missing year_structure.json. Configure and save Year Structure first.")
    if not ac:
        raise RuntimeError("Missing archetype_configs.json. Build archetypes first.")

    seasons = ys.get("seasons", [])
    use_week = bool(ys.get("use_week_classes", False))
    week_classes = ys.get("week_classes", [])

    month_to_season = month_to_season_map(seasons)
    week_pattern = build_week_pattern(week_classes) if use_week else None

    # --- Build pools per archetype & category
    pools: Dict[str, Dict] = {}
    global_categories: set = set()

    for arch_id, meta in ac.items():
        xlsx_path = meta.get("full_excel_path")
        if not xlsx_path:
            continue
        xlsx = Path(xlsx_path)
        if not xlsx.exists():
            continue  # allow not-yet-built rows

        df_full = pd.read_excel(xlsx, sheet_name=0)
        cats = [str(c) for c in df_full.iloc[:, 0].unique()]
        global_categories.update(cats)

        num_days = int(meta.get("num_days", 0))
        per_cat: Dict[str, np.ndarray] = {}

        for cat in cats:
            cat_df = df_full[df_full.iloc[:, 0] == cat]
            buf = BytesIO()
            cat_df.to_excel(buf, index=False)
            buf.seek(0)

            uc = UseCase(name=f"{meta.get('label', arch_id)}__{cat}")
            uc.initialize(num_days=num_days, force=True)
            uc.load(buf)
            daily_profiles = uc.generate_daily_load_profiles(flat=False)
            if daily_profiles is None:
                continue

            arr = np.array(daily_profiles)
            if arr.ndim == 1:
                if arr.size % 1440 != 0:
                    continue
                arr = arr.reshape((-1, 1440))
            per_cat[cat] = arr

        if not per_cat:
            continue

        aggregated_pool = np.sum(list(per_cat.values()), axis=0)  # (num_days_eff, 1440)
        pools[arch_id] = {
            "per_category": per_cat,
            "aggregated": aggregated_pool,
            "num_days": list(per_cat.values())[0].shape[0],
            "label": meta.get("label", arch_id),
            "season": meta.get("season"),
            "week_class": meta.get("week_class"),
        }

    if not pools:
        raise RuntimeError("No archetype pools were generated. Ensure inputs are built.")

    global_categories = sorted(global_categories)

    # --- Build 365-day calendar and assemble (based on 2019 for non-leap year)
    dates = pd.date_range("2019-01-01", periods=365, freq="D")
    rng = default_rng()

    def _arch_id_from_pair(season_name: str, week_class_name: Optional[str]) -> str:
        wc = "none" if not week_class_name else week_class_name
        return f"{season_name}__{wc}".lower().replace(" ", "_").replace("–", "_").replace("-", "_").replace("/", "_").replace("\\", "_")

    year_by_cat: Dict[str, List[np.ndarray]] = {c: [] for c in global_categories}
    year_agg: List[np.ndarray] = []

    # quick reverse index for fallback by season only
    by_season_only = {k for k, v in pools.items() if v.get("week_class") in (None, "none", "")}

    for d in dates:
        season = month_to_season.get(d.month)
        wc = week_pattern[d.weekday()] if week_pattern else None

        aid = _arch_id_from_pair(season, wc)
        if aid not in pools:
            # try season-only
            so = _arch_id_from_pair(season, None)
            if so in pools:
                aid = so
            else:
                aid = next(iter(pools.keys()))  # fallback to first available

        pool = pools[aid]
        per_cat = pool["per_category"]
        agg_pool = pool["aggregated"]
        n = agg_pool.shape[0]
        idx = int(rng.integers(low=0, high=n))

        for cat in global_categories:
            day = per_cat[cat][idx, :] if cat in per_cat else np.zeros(agg_pool.shape[1], dtype=float)
            year_by_cat[cat].append(day)
        year_agg.append(agg_pool[idx, :])

    # Stack & save
    PM.outputs_dir.mkdir(parents=True, exist_ok=True)

    EXPORT_MINUTE_LONG = True

    per_user_csv: Dict[str, str] = {}
    per_user_csv_long: Dict[str, str] = {}

    # --- per-category
    for cat in global_categories:
        mat = np.vstack(year_by_cat[cat])  # (365,1440)

        # 1) Always save WIDE
        wide_path = PM.outputs_dir / f"profile_{_safe_name(cat)}.csv"
        save_dataframe_csv(pd.DataFrame(mat), wide_path)
        per_user_csv[cat] = str(wide_path.resolve())

        # 2) Optionally save LONG
        if EXPORT_MINUTE_LONG:
            df_long = matrix_to_long_series(
                mat, freq="T", year=2019, col_name="power_W", add_datetime_index=False
            )
            long_path = PM.outputs_dir / f"profile_{_safe_name(cat)}_minute_long.csv"
            save_dataframe_csv(df_long, long_path)
            per_user_csv_long[cat] = str(long_path.resolve())

    # --- aggregated
    agg_mat = np.vstack(year_agg)  # (365,1440)

    agg_wide_path = PM.outputs_dir / "profile_aggregated.csv"
    save_dataframe_csv(pd.DataFrame(agg_mat), agg_wide_path)

    agg_long_path = None
    if EXPORT_MINUTE_LONG:
        agg_long = matrix_to_long_series(
            agg_mat, freq="T", year=2019, col_name="power_W", add_datetime_index=False
        )
        agg_long_path = PM.outputs_dir / "profile_aggregated_minute_long.csv"
        save_dataframe_csv(agg_long, agg_long_path)

    return {
        "mode": "multi",
        "days": 365,
        "minutes": 1440,
        "categories": global_categories,
        "aggregated_csv": str(agg_wide_path.resolve()),
        "per_user_csv": per_user_csv,
        "aggregated_csv_long": str(agg_long_path.resolve()) if agg_long_path else None,
        "per_user_csv_long": per_user_csv_long,
    }

