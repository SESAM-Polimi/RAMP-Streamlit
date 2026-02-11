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

from config.path_manager import PM
from core.utils import (
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

        # Save the uploaded FULL template Excel as-is
        PM.archetypes_dir.mkdir(parents=True, exist_ok=True)
        out_path = (PM.archetypes_dir / f"ramp_input_{arch_id}.xlsx").resolve()

        if hasattr(buf, "getvalue"):
            out_path.write_bytes(buf.getvalue())
        else:
            # fallback: if it's a stream, read it
            out_path.write_bytes(buf.read())

        full_path = out_path

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

# ---------------------------------------------------------------------
# SINGLE-ARCHETYPE
# ---------------------------------------------------------------------
def run_single_archetype(excel_path: Path, num_days: int) -> Dict:
    """
    Use ONE full RAMP input file containing all user categories.
    Generates per-category pools (num_days x 1440), then builds a 365-day year
    by randomly sampling days from each pool (with replacement if needed).
    Aggregation is performed AFTER sampling, using the same sampled day index per
    calendar day across all categories for coherence.
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

        # Basic validation
        if arr.ndim != 2 or arr.shape[1] != 1440:
            continue

        per_cat_arrays[cat] = arr

    if not per_cat_arrays:
        raise RuntimeError("No profiles generated. Check the input Excel format.")

    # --- Pooled sampling to 365 days ---
    # Use ONE rng and ONE index sequence so all categories use the same sampled day per calendar day.
    rng = default_rng()
    target_days = 365

    # Determine effective pool length: ideally all categories have same pool length, but don't assume.
    # We'll sample indices per category pool length, but to keep coherence we pick a "reference pool".
    # Choose the smallest pool to avoid out-of-bounds and to keep comparable diversity.
    ref_cat = min(per_cat_arrays.keys(), key=lambda c: per_cat_arrays[c].shape[0])
    ref_pool_days = per_cat_arrays[ref_cat].shape[0]
    if ref_pool_days <= 0:
        raise RuntimeError("Reference category pool is empty.")

    # Sample day indices from reference pool
    # If num_days >= 365 -> sample without replacement gives max diversity; otherwise with replacement.
    replace = not (ref_pool_days >= target_days)
    idx_year = rng.choice(ref_pool_days, size=target_days, replace=replace)

    # Build year per category using the SAME idx_year, clipped to each pool length if needed.
    # (If pools differ in length, we map indices via modulo to keep deterministic behavior.)
    per_cat_year: Dict[str, np.ndarray] = {}
    for cat, mat in per_cat_arrays.items():
        pool_days = mat.shape[0]
        if pool_days <= 0:
            per_cat_year[cat] = np.zeros((target_days, 1440), dtype=float)
            continue

        if pool_days == ref_pool_days:
            idx = idx_year
        else:
            # Map reference indices into this pool size
            idx = idx_year % pool_days

        per_cat_year[cat] = mat[idx, :]

    aggregated = np.sum(list(per_cat_year.values()), axis=0)  # (365, 1440)

    # Save results
    PM.outputs_dir.mkdir(parents=True, exist_ok=True)
    for cat, year_mat in per_cat_year.items():
        df = pd.DataFrame(year_mat)
        save_dataframe_csv(df, PM.outputs_dir / f"profile_{_safe_name(cat)}.csv")

    agg_df = pd.DataFrame(aggregated)
    agg_path = save_dataframe_csv(agg_df, PM.outputs_dir / "profile_aggregated.csv")

    return {
        "mode": "single",
        "num_days": int(aggregated.shape[0]),  # 365
        "minutes": int(aggregated.shape[1]),   # 1440
        "categories": categories,
        "aggregated_csv": str(agg_path),
        "per_user_csv": {cat: str((PM.outputs_dir / f"profile_{_safe_name(cat)}.csv").resolve())
                         for cat in categories},
        "sampling": {
            "method": "pooled_random_sampling",
            "replacement": bool(replace),
            "reference_category": ref_cat,
            "reference_pool_days": int(ref_pool_days),
        },
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
      2) Build a 365-day calendar (2020) using season-by-month + optional week pattern
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

    # --- Build 365-day calendar and assemble
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
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
    per_user_csv: Dict[str, str] = {}
    for cat in global_categories:
        mat = np.vstack(year_by_cat[cat])
        df = pd.DataFrame(mat)
        path = save_dataframe_csv(df, PM.outputs_dir / f"profile_{_safe_name(cat)}.csv")
        per_user_csv[cat] = str(path)

    agg_mat = np.vstack(year_agg)
    agg_df = pd.DataFrame(agg_mat)
    agg_path = save_dataframe_csv(agg_df, PM.outputs_dir / "profile_aggregated.csv")

    return {
        "mode": "multi",
        "days": int(agg_mat.shape[0]),
        "minutes": int(agg_mat.shape[1]),
        "categories": global_categories,
        "aggregated_csv": str(agg_path),
        "per_user_csv": per_user_csv,
    }

