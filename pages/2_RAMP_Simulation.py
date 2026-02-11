import io
import zipfile
from typing import List, Dict

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config.path_manager import PM
from core.utils import (
    clear_directory,
    default_month_partition,
    save_year_structure,
    derive_archetypes,
    load_profiles_from_outputs,
    load_aggregated_from_outputs,
    build_hourly_from_minute,
    save_hourly_aggregated,
    build_calendar_metadata_from_year_structure,
)
from core.ramp_core import run_single_archetype, run_multi_archetype, save_archetype_configs
from core.ramp_schema import RAMPInputSchema

st.title("High-Resolution RAMP Demand Simulation")
st.caption("Configure RAMP seasons, upload inputs, simulate, and visualise load profiles.")

# ---------------------------------------------------------------------------
# Project files & management
# ---------------------------------------------------------------------------
st.subheader("Project files & management")

st.markdown(
    """
    - **Inputs** (full RAMP Excel files, per–archetype inputs) are written under
      the `inputs/` folder of the app.
    - **Outputs** (per–user profiles and aggregated profiles) are written under
      the `outputs/` folder.
    
    It is recommended to:
    - keep your own copies of the Excel inputs you use,
    - use the **Download results** section at the bottom of the page to export
      the generated profiles (ZIP + hourly aggregate) and archive them per
      project / scenario outside this app.
    """
)

if st.button("Clear inputs folder", type="primary"):
    try:
        clear_directory(PM.inputs_dir)
        st.success("Inputs folder cleared (all generated full RAMP inputs and archetype files removed).")
    except Exception as e:
        st.error(f"Could not clear inputs folder: {e}")

if st.button("Clear outputs folder", type="primary"):
    try:
        clear_directory(PM.outputs_dir)
        st.success("Outputs folder cleared (all generated profiles removed).")
    except Exception as e:
        st.error(f"Could not clear outputs folder: {e}")

st.markdown("---")
st.subheader("1. Configure Seasonal & Weekly Structure")
st.markdown(
    """
    This section configures **how the year is structured** before running RAMP.

    **Seasons & Months**
    - Choose how many **seasons** (1–4).
    - For each season: set a **name** and select **calendar months (1–12)** (contiguous or not).

    **Weekly structure**
    - Optionally **differentiate within the week** (e.g., *Weekday* vs *Weekend*).
    - For each class, set a **name** and **days per week** (sum ideally **7**; otherwise used as relative frequencies).
    """
)

# -----------------------------------------------------------------------------
# Session defaults
# -----------------------------------------------------------------------------
if "ys" not in st.session_state:
    st.session_state.ys = {
        "n_seasons": 1,
        "seasons": [{"name": "Season 1", "months": list(range(1, 13))}],
        "use_week_classes": False,
        "week_classes": []  # [{"name":"Weekday","days_per_week":5}, {"name":"Weekend","days_per_week":2}]
    }
ys = st.session_state.ys

if "inputs_updated" not in st.session_state:
    st.session_state.inputs_updated = False

if "inputs_built" not in st.session_state:
    st.session_state.inputs_built = False

if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False

# -----------------------------------------------------------------------------
# Seasons editor
# -----------------------------------------------------------------------------
with st.container():
    # Previous number of seasons (for change detection)
    prev_n = st.session_state.get("prev_n_seasons", int(ys["n_seasons"]))

    n_seasons = st.number_input(
        "Number of seasons",
        min_value=1,
        max_value=4,
        value=int(ys["n_seasons"]),
        step=1,
    )
    n_seasons = int(n_seasons)
    ys["n_seasons"] = n_seasons
    st.session_state.n_seasons = n_seasons

    # If the number of seasons changed, re-initialise with clean defaults
    if n_seasons != prev_n:
        default_parts = default_month_partition(n_seasons)

        if n_seasons == 1:
            # Single season: fixed label "Year" and full year
            ys["seasons"] = [{
                "name": "Year",
                "months": list(range(1, 13)),
            }]
        else:
            # Multi-season: default labels "Season 1", "Season 2", ...
            # and evenly split months from default_month_partition
            ys["seasons"] = [
                {
                    "name": f"Season {i+1}",
                    "months": default_parts[i],
                }
                for i in range(n_seasons)
            ]
    else:
        # Same number of seasons as before → preserve existing structure,
        # only resize list if needed (e.g. app reload with missing entries).
        if n_seasons > len(ys["seasons"]):
            for i in range(len(ys["seasons"]), n_seasons):
                ys["seasons"].append({
                    "name": f"Season {i+1}",
                    "months": [],
                })
        elif n_seasons < len(ys["seasons"]):
            ys["seasons"] = ys["seasons"][:n_seasons]

    # Store for next run (needed for change detection above)
    st.session_state.prev_n_seasons = n_seasons

    # Default month partition (used only as fallback inside widgets)
    default_parts = default_month_partition(n_seasons)

    if n_seasons == 1:
        # Always enforce full-year + label "Year" for single-season mode
        ys["seasons"][0] = {
            "name": "Year",
            "months": list(range(1, 13)),
        }
        st.session_state.seasons = ys["seasons"]

        st.success("Single season selected: all months (1–12) are assigned to this season.")
    else:
        # Multi-season UX: show month allocation widgets
        for i in range(n_seasons):
            with st.expander(f"Season {i+1}", expanded=False):
                s_name = st.text_input(
                    "Name",
                    value=ys["seasons"][i]["name"],
                    key=f"s_name_{i}",
                )
                # If no months are defined yet, fall back to the even split
                current = ys["seasons"][i]["months"] or default_parts[i]
                months = st.multiselect(
                    "Months (1–12)",
                    options=list(range(1, 13)),
                    default=current,
                    format_func=lambda m: [
                        "Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"
                    ][m-1],
                    key=f"s_months_{i}",
                )
                ys["seasons"][i] = {
                    "name": s_name.strip() or f"Season {i+1}",
                    "months": sorted(months),
                }

        st.session_state.seasons = ys["seasons"]


# -----------------------------------------------------------------------------
# Weekly structure editor
# -----------------------------------------------------------------------------
with st.container():
    use_week = st.checkbox(
        "Differentiate within week?",
        value=ys.get("use_week_classes", False),
        key="use_week_classes_cb"
    )
    ys["use_week_classes"] = bool(use_week)
    st.session_state.use_week_classes = ys["use_week_classes"]

    week_classes: List[Dict] = ys.get("week_classes", []) or []
    if ys["use_week_classes"]:
        default_n = max(2, len(week_classes)) if week_classes else 2
        n_classes = st.number_input(
            "Number of week-classes",
            min_value=1, max_value=7,
            value=default_n, step=1,
            key="n_week_classes"
        )

        # Resize while preserving existing entries
        curr = week_classes[:]
        if n_classes > len(curr):
            for i in range(len(curr), n_classes):
                if n_classes == 2 and i in (0, 1):
                    label = "Weekday" if i == 0 else "Weekend"
                    days = 5 if i == 0 else 2
                else:
                    label = f"Class {i+1}"
                    days = 1
                curr.append({"name": label, "days_per_week": days})
        else:
            curr = curr[:n_classes]

        # Render each class row
        new_week_classes: List[Dict] = []
        for i in range(n_classes):
            c1, c2 = st.columns([3, 1])
            cname = c1.text_input(
                f"Class {i+1} name",
                value=curr[i]["name"],
                key=f"wc_name_{i}"
            )
            days = c2.number_input(
                "Days/week",
                min_value=1, max_value=7,
                value=int(curr[i]["days_per_week"]),
                key=f"wc_days_{i}"
            )
            new_week_classes.append({
                "name": cname.strip() or f"Class {i+1}",
                "days_per_week": int(days)
            })

        week_classes = new_week_classes
    else:
        week_classes = []

    ys["week_classes"] = week_classes
    st.session_state.week_classes = ys["week_classes"]

# -----------------------------------------------------------------------------
# Validation & Summary
# -----------------------------------------------------------------------------
with st.container():
    if ys["n_seasons"] == 1:
        # No month-partition validation needed in single-season mode
        st.success("Single season selected: all months (1–12) are assigned to this season.")
    else:
        # Validate month coverage only when multiple seasons are used
        all_assigned = [m for s in ys["seasons"] for m in s["months"]]
        all_unique = sorted(set(all_assigned))
        full_year = list(range(1, 13))

        if all_unique == full_year and len(all_assigned) == len(all_unique):
            st.success("Each month 1–12 is assigned exactly once across seasons (no overlaps).")
        else:
            missing = [m for m in full_year if m not in all_unique]
            overlaps = sorted({m for m in full_year if all_assigned.count(m) > 1})
            if missing:
                st.warning(f"Some months are not assigned: {', '.join(map(str, missing))}")
            if overlaps:
                st.warning(f"Some months are assigned to multiple seasons: {', '.join(map(str, overlaps))}")
            st.info("Tip: for a clean partition, assign each month exactly once.")

    # Persist snapshot
    st.session_state.ys = ys

def validate_and_save_ramp_excel(
    uploaded_file,
    out_path,
    *,
    strict_unknown_cols: bool = False,
    fill_missing_optional: bool = True,
    preview_rows: int = 15,
):
    """
    Read uploaded Excel -> map pretty/canonical headers -> fill defaults -> save canonical Excel to out_path.
    Returns (df_canon, report).
    """
    schema = RAMPInputSchema()

    raw_bytes = uploaded_file.getvalue()
    df_raw = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=0)

    df_canon, report = schema.to_canonical(
        df_raw,
        strict_unknown_cols=strict_unknown_cols,
        fill_missing_optional=fill_missing_optional,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as canonical excel (single sheet)
    df_canon.to_excel(out_path, index=False)

    # UI feedback (optional)
    st.success(f"Input validated and saved to: `{out_path}`")
    if report.used_pretty_headers:
        with st.expander("Header mapping report", expanded=False):
            st.write("**Pretty → canonical columns used:**")
            st.json(report.used_pretty_headers)
            if report.ignored_columns:
                st.warning("Ignored unknown columns: " + ", ".join(report.ignored_columns))
            if report.filled_defaults:
                st.info(f"Filled missing optional columns with defaults: {len(report.filled_defaults)}")
            if report.warnings:
                for w in report.warnings:
                    st.warning(w)

    st.dataframe(df_canon.head(preview_rows), use_container_width=True)
    return df_canon, report

# -----------------------------------------------------------------------------
# Actions: Save config + (Single-archetype) upload & build full input
# -----------------------------------------------------------------------------

save_clicked = st.button("💾 Save Year Structure")
if save_clicked:
    try:
        save_year_structure(ys, as_yaml=True)
        st.success("Year structure saved.")
        st.session_state.inputs_updated = True
    except Exception as e:
        st.error(f"Save failed: {e}")

# Single-archetype detection
is_single_archetype = (ys["n_seasons"] == 1 and not ys["use_week_classes"])

st.markdown("---")
st.subheader("2. Upload RAMP Inputs File")
st.markdown(
    """
    The uploaded file should follow the standard *template format*: the **first column contains the user category**, and the following columns contain
    the **appliance definitions** and their required parameters (power, usage window, stochastic settings, etc.). Each row corresponds to *one appliance* of a given category. The app will automatically split the file by category and generate the corresponding
    minute-resolution daily profiles.

    A ready-to-use **example template** is provided in  
    `examples/RAMP Excel Inputs - Example.xlsx`, you can copy and adapt it for your project.
    """
)
st.image(PM.assets_dir / "ramp_template.png", use_container_width=True, caption="Excerpt from the RAMP Excel template")


if st.session_state.inputs_updated:
    if is_single_archetype:
        st.caption("Single-archetype detected (1 season, no weekly differentiation).")
        st.markdown("**Upload the full RAMP Excel file (template format).**")

        uploaded = st.file_uploader(
            "Upload RAMP Excel Input file",
            type=["xlsx", "xls"],
            accept_multiple_files=False
        )

        if uploaded is not None:
            try:
                df_full, report = validate_and_save_ramp_excel(
                    uploaded,
                    PM.full_input_xlsx,
                    strict_unknown_cols=False,     # set True if you want to hard-fail on unknown headers
                    fill_missing_optional=True,    # “autofill missing columns” behavior
                    preview_rows=15,
                )
                st.session_state.inputs_built = True
            except Exception as e:
                st.error(f"Upload/validation failed: {e}")
    else:
        st.caption("Multi-archetype detected (multiple seasons and/or weekly differentiation).")
        st.markdown(
            """
            For every archetype (**Season × Week-class** or just Season), RAMP generates
            `num_days` **independent synthetic daily profiles** at minute resolution.
            In **multi-archetype mode**, the final **365-day year** is built by walking
            through a calendar year. For each calendar day the app:
                1. Determines its season and, if enabled, its week-class.
                2. Selects the corresponding archetype.
                3. Randomly picks **one** day from that archetype’s `num_days` pool
                (for each user category) and uses it as the load shape for that date.

            **Practical guidance**

            - With **1 season and no weekly differentiation** (single-archetype mode),
            `num_days` is the number of stochastic days you ask RAMP to generate;
            these are then tiled or downsampled internally to obtain a full 365-day year.
            - With **multiple archetypes**, each archetype has its own `num_days`
            (its own pool of synthetic days). The yearly profile is assembled by
            sampling from these pools according to the season/week-class calendar.
            """
        )

        st.markdown("**For each archetype, upload the RAMP Excel file and set `num_days`.**")

        archetypes = derive_archetypes(ys)  # [{"season","week_class","label","arch_id"}, ...]
        if not archetypes:
            st.info("No archetypes generated from the current year structure. "
                    "Check seasons and weekly settings above.")
        else:
            st.caption("One row per archetype (Season × Week-class or just Season). "
                    "Choose `num_days` and upload the simplified Excel for each row you want to build.")

            # --- Keep per-row states in session, but ONLY for active archetypes ---
            active_ids = {a["arch_id"] for a in archetypes}

            if "arch_rows" not in st.session_state:
                # First time: create rows only for current archetypes
                st.session_state.arch_rows = {
                    a["arch_id"]: {
                        "season": a["season"],
                        "week_class": a["week_class"],
                        "label": a["label"],
                        "num_days": 60,
                        "file_like": None,
                    }
                    for a in archetypes
                }
            else:
                # 1) Drop rows that no longer correspond to any active archetype
                existing_rows = st.session_state.arch_rows
                pruned_rows = {
                    arch_id: meta
                    for arch_id, meta in existing_rows.items()
                    if arch_id in active_ids
                }

                # 2) Add rows for newly created archetypes (if user changed config)
                for a in archetypes:
                    if a["arch_id"] not in pruned_rows:
                        pruned_rows[a["arch_id"]] = {
                            "season": a["season"],
                            "week_class": a["week_class"],
                            "label": a["label"],
                            "num_days": 60,
                            "file_like": None,
                        }

                st.session_state.arch_rows = pruned_rows

            rows = st.session_state.arch_rows

            # editable form
            with st.form("archetype_form", clear_on_submit=False):
                for a in archetypes:
                    arch_id = a["arch_id"]
                    meta = rows[arch_id]

                    with st.expander(meta["label"], expanded=False):
                        col1, col2 = st.columns([1, 2])
                        meta["num_days"] = int(col1.number_input(
                            "num_days", min_value=1, max_value=365, value=int(meta["num_days"]),
                            key=f"numdays_{arch_id}"
                        ))
                        uploaded = col2.file_uploader(
                            "Upload RAMP Excel (.xlsx/.xls)", type=["xlsx", "xls"],
                            accept_multiple_files=False, key=f"file_{arch_id}"
                        )
                        # Store bytes if provided
                        if uploaded is not None:
                            meta["file_like"] = io.BytesIO(uploaded.getvalue())

                        # read-only info
                        r1, r2, r3 = st.columns(3)
                        r1.write(f"**Season:** {meta['season']}")
                        r2.write(f"**Week-class:** {meta['week_class'] if meta['week_class'] else '—'}")
                        r3.write(f"`arch_id`: `{arch_id}`")

                submit = st.form_submit_button("Build inputs for selected archetypes")

                if submit:
                    try:
                        schema = RAMPInputSchema()

                        built = {}  # mimic your previous status dict
                        configs_to_save = {}

                        for a in archetypes:
                            arch_id = a["arch_id"]
                            meta = rows[arch_id]

                            file_like = meta.get("file_like", None)
                            if file_like is None:
                                built[arch_id] = {
                                    "label": meta.get("label", ""),
                                    "num_days": meta.get("num_days", ""),
                                    "error": "missing upload",
                                }
                                continue

                            # Read user file
                            df_raw = pd.read_excel(file_like, sheet_name=0)

                            # Schema -> canonical
                            df_canon, report = schema.to_canonical(
                                df_raw,
                                strict_unknown_cols=False,
                                fill_missing_optional=True,
                            )

                            # Save canonical archetype Excel
                            out_path = PM.archetypes_dir / f"ramp_input_{arch_id}.xlsx"
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            df_canon.to_excel(out_path, index=False)

                            built[arch_id] = {
                                "label": meta.get("label", ""),
                                "num_days": int(meta.get("num_days", 60)),
                                "full_excel_path": str(out_path),
                            }
                            configs_to_save[arch_id] = {
                                "season": meta.get("season"),
                                "week_class": meta.get("week_class"),
                                "label": meta.get("label"),
                                "num_days": int(meta.get("num_days", 60)),
                                "full_excel_path": str(out_path),
                            }

                        # Persist archetype configs for ramp_core
                        save_archetype_configs(configs_to_save)

                        # Status table
                        status_records = []
                        for arch_id, info in built.items():
                            status_records.append({
                                "arch_id": arch_id,
                                "label": info.get("label", ""),
                                "num_days": info.get("num_days", ""),
                                "built": "yes" if "full_excel_path" in info else f"no ({info.get('error','')})",
                                "full_excel_path": info.get("full_excel_path", ""),
                            })

                        st.success("Archetype inputs validated and saved.")
                        st.dataframe(pd.DataFrame(status_records), use_container_width=True)
                        st.caption("Archetype metadata saved to `config/archetype_configs.json`.")
                        st.session_state.inputs_built = True

                    except Exception as e:
                        st.error(f"Archetype build failed: {e}")
else:
    st.info("Year Structure not saved or not changed yet.")

# --- Run Simulation -----------------------------------------------------------
st.markdown("---")
st.subheader("Run RAMP simulation")

if st.session_state.inputs_built:
    ys = st.session_state.get("ys", {})
    is_single = (ys.get("n_seasons", 1) == 1 and not ys.get("use_week_classes", False))

    if is_single:
        st.caption("Single-archetype mode detected (1 season, no weekly differentiation).")
        num_days = st.number_input("num_days (single archetype)", 1, 365, 365, step=1)
        if num_days < 365:
            st.warning(
                f"You selected {num_days} day(s). "
                "The app will randomly resample from this pool to assemble a full 365-day year "
                "(sampling with replacement if needed). "
                "This speeds up computation but may repeat similar day patterns and reduce overall variability."
            )
        run_single = st.button("🚀 Run single-archetype simulation")

        if run_single:
            with st.spinner("Running simulation..."):
                try:
                    excel_path = PM.full_input_xlsx
                    summary = run_single_archetype(excel_path, int(num_days))
                    st.success("Simulation completed (single-archetype).")
                    st.session_state.simulation_done = True
                except Exception as e:
                    st.error(f"Single-archetype run failed: {e}")

    else:
        st.caption("Multi-archetype mode detected (season and/or week-class differentiation).")
        run_multi = st.button("🚀 Run multi-archetype simulation")

        if run_multi:
            with st.spinner("Running simulation..."):
                try:
                    summary = run_multi_archetype()
                    st.success("Simulation completed (multi-archetype).")
                    st.session_state.simulation_done = True
                except Exception as e:
                    st.error(f"Multi-archetype run failed: {e}")
else:
    st.info("Inputs not yet built. Cannot run simulation.")

# --- Daily Average Load Curve (Minute Resolution) -----------------------------
st.markdown("---")
st.subheader("Visualize Load Profiles")

st.markdown("**Daily Average Load Curve (Minute Resolution)**")
if not st.session_state.simulation_done:
    st.info("No simulation done yet. Run a simulation to visualize profiles.")
else:
    # Load from disk (works whether you just ran a sim or are reopening the app)
    users_list, aggregated_matrix = load_profiles_from_outputs()

    # Build options: Aggregated first (if available), then user names
    names = [u.name for u in users_list]
    aggregated_available = aggregated_matrix is not None
    if aggregated_available:
        names = ["Aggregated"] + names

    if not names:
        st.info("No profiles found yet. Run a simulation first.")
    else:
        # Default to Aggregated if available, otherwise first user
        default_index = 0
        selected = st.selectbox("Select profile to visualize", names, index=default_index)


        # Build the 2D matrix (days x 1440)
        if selected == "Aggregated":
            profiles = aggregated_matrix
            title_base = "Aggregated load profile"
        else:
            sel = next((u for u in users_list if u.name == selected), None)
            if sel is None:
                st.warning(f"Could not find user '{selected}'.")
                st.stop()
            profiles = sel.demand_data.to_numpy()
            title_base = f"Load profile — {selected}"

        # Validate & reshape to (n_days, 1440)
        minutes = 1440
        profiles = np.asarray(profiles)
        if profiles.ndim == 1:
            if profiles.size % minutes != 0:
                st.error(f"Profile length {profiles.size} is not a multiple of 1440.")
                st.stop()
            n_days = profiles.size // minutes
            mat = profiles.reshape((n_days, minutes))
        else:
            n_days, n_minutes = profiles.shape
            mat = profiles
            if n_minutes != minutes:
                st.warning(f"Each day has {n_minutes} minutes instead of 1440.")

        # --- Scope controls (season / week-class filtering if present) ---
        ys_local = st.session_state.get("ys", {})
        allow_season = int(ys_local.get("n_seasons", 1)) > 1
        allow_week   = bool(ys_local.get("use_week_classes", False))

        # Derive calendar metadata when needed (length must match n_days)
        dates = seasons_for_days = weekclasses_for_days = None
        if allow_season or allow_week:
            dates, seasons_for_days, weekclasses_for_days = build_calendar_metadata_from_year_structure(mat.shape[0])

        scope_options = ["Whole year"]
        if allow_season and seasons_for_days is not None:
            scope_options.append("By season")
        if allow_week and weekclasses_for_days is not None and any(weekclasses_for_days):
            scope_options.append("By week class")

        scope = st.selectbox("Scope", scope_options, index=0)

        # Resolve indices for the selected scope
        idx = np.arange(mat.shape[0])
        scope_label = ""
        if scope == "By season" and seasons_for_days is not None:
            unique_seasons = list(pd.Series(seasons_for_days).dropna().unique())
            chosen_season = st.selectbox("Season", unique_seasons, index=0)
            idx = np.where(np.array(seasons_for_days) == chosen_season)[0]
            scope_label = f" — {chosen_season}"
            if idx.size == 0:
                st.warning("No days matched this season. Showing whole year instead.")
                idx = np.arange(mat.shape[0])
                scope = "Whole year"
                scope_label = ""
        elif scope == "By week class" and weekclasses_for_days is not None:
            unique_wc = [w for w in pd.Series(weekclasses_for_days).dropna().unique().tolist() if w]
            chosen_wc = st.selectbox("Week class", unique_wc, index=0)
            idx = np.where(np.array(weekclasses_for_days) == chosen_wc)[0]
            scope_label = f" — {chosen_wc}"
            if idx.size == 0:
                st.warning("No days matched this week class. Showing whole year instead.")
                idx = np.arange(mat.shape[0])
                scope = "Whole year"
                scope_label = ""

        # Subset matrix by scope
        mat_scoped = mat[idx, :] if idx.size and idx.size != mat.shape[0] else mat

        # Compute bands & average (always enabled)
        prof_min = mat_scoped.min(axis=0)
        prof_max = mat_scoped.max(axis=0)
        prof_avg = mat_scoped.mean(axis=0)

        factor = 1000.0  # always display in kW
        y_label = "Power [kW]"

        # Plot average daily + band + cloud (always on)
        fig, ax = plt.subplots(figsize=(14, 6))

        # cloud
        for day in mat_scoped:
            ax.plot(day / factor, linewidth=0.4, alpha=0.25, color="gray")

        # min–max band
        ax.fill_between(
            range(minutes),
            prof_min / factor,
            prof_max / factor,
            alpha=0.25,
            color="lightgray",
            label="Variability range",
        )

        # average
        ax.plot(prof_avg / factor, linewidth=2.0, color="red", label="Average daily profile")

        ax.set_title(f"{title_base}{scope_label}")
        ax.set_xlabel("Time of day")
        ax.set_ylabel(y_label)
        ax.set_xticks(np.linspace(0, minutes, 24))
        ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        st.pyplot(fig)

        # --- Second plot: inspect a specific day (slider 1..n_days) ---
        st.markdown(f"**Inspect a specific day ({selected})**")
        day_idx = st.slider("Day of year", min_value=1, max_value=mat.shape[0], value=1, step=1)
        day_vec = mat[day_idx - 1, :]  # 1-based → 0-based

        fig2, ax2 = plt.subplots(figsize=(14, 4))
        ax2.plot(day_vec / factor, linewidth=1.8)
        ax2.set_title(f"Day #{day_idx} — {title_base}")
        ax2.set_xlabel("Time of day")
        ax2.set_ylabel(y_label)
        ax2.set_xticks(np.linspace(0, minutes, 24))
        ax2.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

# --- Average Hourly Load (Aggregated) ----------------------------------------
st.markdown("---")
st.markdown("**Average Hourly Load (Aggregated)**")

# Try to reuse aggregated_matrix from above; else load from outputs
agg_minute = None
if "aggregated_matrix" in locals() and isinstance(aggregated_matrix, np.ndarray):
    agg_minute = aggregated_matrix
else:
    agg_minute = load_aggregated_from_outputs()

if agg_minute is None:
    st.info("No aggregated profile found yet. Run a simulation to create `/outputs/profile_aggregated.csv`.")
else:
    try:
        hourly = build_hourly_from_minute(agg_minute)
    except Exception as e:
        st.error(f"Could not build hourly series: {e}")
        st.stop()

    # Save hourly series
    out_path = save_hourly_aggregated(hourly)
    st.caption(f"Hourly aggregated profile saved to: `{out_path}`")

    # Compute stats for plot (kW)
    mean_hourly = hourly.mean(axis=0) / 1000.0
    min_hourly  = hourly.min(axis=0) / 1000.0
    max_hourly  = hourly.max(axis=0) / 1000.0

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mean_hourly, linewidth=2, label="Average hourly load")
    ax.fill_between(range(24), min_hourly, max_hourly, alpha=0.2, label="Min–max range")
    ax.set_title("Average daily hourly load profile (aggregated)")
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Power [kW]")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # ------------------------------------------------------------------
    # Downloads: aggregated minute profile, all outputs, hourly profile
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Download results")
    st.markdown(
        """
        Export the demand profiles generated by RAMP for further analysis.
        - The **aggregated minute profile (365 × 1440)** is the full high-resolution load curve.
        - The **ZIP of all profiles** contains individual user-category profiles as well as aggregated.
        - The **aggregated hourly profile (8760)** is recommended for energy system optimization, dispatch, and planning models.
        """
    )

    # 1) Aggregated minute-resolution profile (365 × 1440)
    agg_csv_path = PM.outputs_dir / "profile_aggregated.csv"
    if agg_csv_path.exists():
        with open(agg_csv_path, "rb") as f:
            st.download_button(
                "Download aggregated minute profile (365×1440)",
                data=f.read(),
                file_name="profile_aggregated.csv",
                mime="text/csv",
            )

    # 2) All profiles in outputs/ as a ZIP (per-user + aggregated)
    if PM.outputs_dir.exists():
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in PM.outputs_dir.glob("*.csv"):
                zf.write(p, arcname=p.name)
        zip_buf.seek(0)
        st.download_button(
            "Download all profiles (outputs folder, ZIP)",
            data=zip_buf,
            file_name="ramp_outputs_profiles.zip",
            mime="application/zip",
        )

    # 3) Aggregated hourly profile (8760 hours)
    if out_path.exists():
        with open(out_path, "rb") as f:
            st.download_button(
                "Download aggregated hourly profile (8760 hours)",
                data=f.read(),
                file_name="profile_aggregated_hourly.csv",
                mime="text/csv",
            )

