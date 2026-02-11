import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

from config.path_manager import PM
from core.ramp_schema import PRETTY_TO_CANON

# Inverse mapping for UI (canonical -> pretty)
CANON_TO_PRETTY = {v: k for k, v in PRETTY_TO_CANON.items()}

# Your editor uses w1_start_min etc (not part of RAMP canonical template)
# Add pretty labels for those editor-only columns:
EDITOR_ONLY_PRETTY = {
    "w1_start_min": "FW1 start [min]",
    "w1_end_min":   "FW1 end [min]",
    "w2_start_min": "FW2 start [min]",
    "w2_end_min":   "FW2 end [min]",
    "w3_start_min": "FW3 start [min]",
    "w3_end_min":   "FW3 end [min]",
}

# canonical->pretty for editor columns
UI_PRETTY_MAP = {**CANON_TO_PRETTY, **EDITOR_ONLY_PRETTY}

# pretty->canonical for editor columns (for back-mapping after edit)
UI_CANON_MAP = {v: k for k, v in UI_PRETTY_MAP.items()}


def df_to_pretty(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Take canonical df, return pretty-header df with only `cols`."""
    view = df.reindex(columns=cols).copy()
    return view.rename(columns=UI_PRETTY_MAP)


def df_to_canon(df_pretty: pd.DataFrame) -> pd.DataFrame:
    """Take pretty-header df from editor, map back to canonical headers."""
    out = df_pretty.copy()
    out.columns = [UI_CANON_MAP.get(c, c) for c in out.columns]
    return out

# ----------------------------
# Helpers for UI defaults
# ----------------------------
STD_COLS = [
    "name", "number", "power",
    "num_windows", "func_time",
    "time_fraction_random_variability", "func_cycle",
    "occasional_use",
    "flat", "fixed",
    "thermal_p_var", "pref_index", "wd_we_type",
    # windows in minutes (0..1440)
    "w1_start_min", "w1_end_min",
    "w2_start_min", "w2_end_min",
    "w3_start_min", "w3_end_min",
]

DUTY_COLS = [
    "name", "number",
    "fixed_cycle",
    # cycle segments (show a minimal subset; you can expand later)
    "p_11", "t_11", "cw11_start", "cw11_end", "r_c1",
    "p_21", "t_21", "cw21_start", "cw21_end", "r_c2",
    "p_31", "t_31", "cw31_start", "cw31_end", "r_c3",
    # still include general knobs often relevant
    "occasional_use", "flat", "fixed", "thermal_p_var", "pref_index", "wd_we_type",
    # base windows (optional, but many duty appliances still have full-day window)
    "w1_start_min", "w1_end_min",
]

STD_PRETTY_COLS = [UI_PRETTY_MAP.get(c, c) for c in STD_COLS]
DUTY_PRETTY_COLS = [UI_PRETTY_MAP.get(c, c) for c in DUTY_COLS]

STD_EDITOR_CONFIG = {
    UI_PRETTY_MAP["flat"]:  st.column_config.SelectboxColumn("Flat profile within windows? [yes/no]", options=["yes", "no"]),
    UI_PRETTY_MAP["fixed"]: st.column_config.SelectboxColumn("Fixed daily schedule? [yes/no]", options=["yes", "no"]),
}

DUTY_EDITOR_CONFIG = {
    UI_PRETTY_MAP["flat"]:  st.column_config.SelectboxColumn("Flat profile within windows? [yes/no]", options=["yes", "no"]),
    UI_PRETTY_MAP["fixed"]: st.column_config.SelectboxColumn("Fixed daily schedule? [yes/no]", options=["yes", "no"]),
}

def _default_std_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(
            name="Indoor Bulb", number=3, power=7,
            num_windows=2, func_time=240,
            time_fraction_random_variability=0.2, func_cycle=5,
            occasional_use=1.0,
            flat="no", fixed="no",
            thermal_p_var=0, pref_index=0, wd_we_type=2,
            w1_start_min=18*60, w1_end_min=24*60,
            w2_start_min=6*60,  w2_end_min=int(7.5*60),
            w3_start_min=0,     w3_end_min=0,
        )
    ], columns=STD_COLS)

def _default_duty_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(
            name="Fridge", number=1, fixed_cycle=3,
            p_11=150, t_11=20, cw11_start=580, cw11_end=1200, r_c1=0.0,
            p_21=150, t_21=15, cw21_start=420, cw21_end=579,  r_c2=0.0,
            p_31=150, t_31=10, cw31_start=0,   cw31_end=419,  r_c3=0.0,
            occasional_use=1.0, flat="no", fixed="yes",
            thermal_p_var=0, pref_index=0, wd_we_type=2,
            w1_start_min=0, w1_end_min=1440,
        )
    ], columns=DUTY_COLS)

def _clamp_minute(x):
    try:
        v = int(float(x))
    except Exception:
        return 0
    return int(max(0, min(1440, v)))

def _windows_to_hour_mask(row) -> np.ndarray:
    """
    Returns 24-length mask (0/1) of which hours are covered by the union of up to 3 windows.
    Windows are given in minutes [0..1440]. End==1440 allowed.
    """
    mask = np.zeros(24, dtype=int)

    for i in (1, 2, 3):
        s = _clamp_minute(row.get(f"w{i}_start_min", 0))
        e = _clamp_minute(row.get(f"w{i}_end_min", 0))

        if s == 0 and e == 0:
            continue
        if e < s:
            # treat as invalid for v1; could also interpret as wrapping over midnight
            continue

        # mark any hour that overlaps [s, e)
        for h in range(24):
            hs = h * 60
            he = (h + 1) * 60
            if max(hs, s) < min(he, e):
                mask[h] = 1
    return mask

def _plot_windows_heatmap(all_apps: pd.DataFrame, title: str):
    """
    Plot a simple heatmap where rows=appliances, cols=hours.
    """
    # Keep only rows with a name
    df = all_apps.copy()
    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["name"] != ""]
    if df.empty:
        st.info("Add appliances to visualize their time windows.")
        return

    mat = np.vstack([_windows_to_hour_mask(r) for _, r in df.iterrows()])
    labels = df["name"].tolist()

    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.35 * len(labels))))
    ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(False)
    st.pyplot(fig)

# ---- Template loader  ----
TEMPLATE_PATH = PM.template_file  

@st.cache_data
def load_template_df(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def pick_base_template_row(template_df: pd.DataFrame, app_name: str, user_name: str) -> pd.Series:
    """
    Choose the best base row from the template to inherit defaults.
    Priority:
      1) same appliance name + same user_name
      2) same appliance name
      3) first row as fallback
    """
    app_name = str(app_name).strip()
    user_name = str(user_name).strip()

    if "name" in template_df.columns and "user_name" in template_df.columns:
        base = template_df[(template_df["name"] == app_name) & (template_df["user_name"] == user_name)]
        if not base.empty:
            return base.iloc[0].copy()
        base = template_df[(template_df["name"] == app_name)]
        if not base.empty:
            return base.iloc[0].copy()

    return template_df.iloc[0].copy()

def _ensure_col(row: pd.Series, col: str, default=0):
    if col not in row.index:
        row[col] = default

def compile_full_ramp_df(categories: dict, template_df: pd.DataFrame) -> pd.DataFrame:
    """
    categories: st.session_state.categories structure
    template_df: the full RAMP template dataframe (authoritative columns)
    returns: full_df with EXACT template columns
    """
    out_rows = []

    # map editor minutes -> template window cols
    win_map = {
        "w1_start_min": "window_1_start",
        "w1_end_min":   "window_1_end",
        "w2_start_min": "window_2_start",
        "w2_end_min":   "window_2_end",
        "w3_start_min": "window_3_start",
        "w3_end_min":   "window_3_end",
    }

    for user_name, meta in categories.items():
        num_users = int(meta.get("num_users", 1))
        user_pref = int(meta.get("user_preference", 1))

        std_df = meta.get("std_df", pd.DataFrame()).copy()
        duty_df = meta.get("duty_df", pd.DataFrame()).copy()

        # ---- Standard appliances ----
        for _, a in std_df.iterrows():
            app_name = str(a.get("name", "")).strip()
            if not app_name:
                continue

            row = pick_base_template_row(template_df, app_name, user_name)

            # user-level
            row["user_name"] = user_name
            _ensure_col(row, "num_users", 1)
            row["num_users"] = num_users
            if "user_preference" in row.index:
                row["user_preference"] = user_pref  # repeated in each row (simple + robust)

            # standard appliance fields
            for col in ["name", "number", "power", "num_windows", "func_time",
                        "time_fraction_random_variability", "func_cycle",
                        "occasional_use", "flat", "fixed",
                        "thermal_p_var", "pref_index", "wd_we_type"]:
                if col in row.index and col in a.index:
                    row[col] = a[col]

            # if template expects random_var_w, set it (safe default: same as time_fraction_random_variability)
            if "random_var_w" in row.index:
                tv = a.get("time_fraction_random_variability", 0)
                row["random_var_w"] = tv

            # windows
            for editor_col, tpl_col in win_map.items():
                if tpl_col in row.index:
                    row[tpl_col] = int(a.get(editor_col, 0) or 0)

            # duty-cycle columns: make them safe defaults (if present)
            # (so standard appliances won't break template constraints)
            duty_like_cols = [c for c in template_df.columns if c.startswith(("p_", "t_", "cw", "r_c", "fixed_cycle"))]
            for c in duty_like_cols:
                if c in row.index and pd.isna(row[c]):
                    row[c] = 0

            out_rows.append(row)

        # ---- Duty-cycle appliances ----
        for _, a in duty_df.iterrows():
            app_name = str(a.get("name", "")).strip()
            if not app_name:
                continue

            row = pick_base_template_row(template_df, app_name, user_name)

            # user-level
            row["user_name"] = user_name
            _ensure_col(row, "num_users", 1)
            row["num_users"] = num_users
            if "user_preference" in row.index:
                row["user_preference"] = user_pref

            # duty-cycle + common knobs
            for col in a.index:
                if col in row.index:
                    row[col] = a[col]

            # windows (some duty-cycle appliances still use a base admissible window)
            for editor_col, tpl_col in win_map.items():
                if tpl_col in row.index:
                    # duty table only has w1_* in your current design; others default to 0
                    row[tpl_col] = int(a.get(editor_col, 0) or 0)

            # if standard fields exist but missing, set safe defaults
            if "num_windows" in row.index and (pd.isna(row["num_windows"]) or row["num_windows"] == ""):
                row["num_windows"] = 1
            if "func_time" in row.index and (pd.isna(row["func_time"]) or row["func_time"] == ""):
                row["func_time"] = 1440

            out_rows.append(row)

    if not out_rows:
        return template_df.iloc[0:0].copy()

    full_df = pd.DataFrame(out_rows)

    # enforce template columns and order
    for c in template_df.columns:
        if c not in full_df.columns:
            full_df[c] = 0
    full_df = full_df[template_df.columns]

    return full_df

# ----------------------------
# Category editor (multi)
# ----------------------------
st.title("RAMP Inputs Builder")
st.caption("Create and validate full RAMP Excel inputs directly in the app, then download the ready-to-run file.")

st.markdown(
    """
This page helps you build **full RAMP input Excel files** without manually editing the large template.

1. Use the **Category editor** to define user categories (e.g., Households, Clinic Tier 1…).
2. For each category, add **standard appliances** and/or **duty-cycle appliances**.
3. Validate inputs visually using the **time-window heatmap** (and quick checks).
4. Click **Build & Download full RAMP Excel** to export a file that matches the original RAMP template.
5. Go to the **Simulation page** and upload the exported Excel file to run RAMP.

**Notes**
- Time windows are expressed in **minutes (0–1440)** on this page for robustness and copy/paste friendliness.
- Appliances entered as *standard* will have duty-cycle columns filled with safe defaults (0/empty).
- Appliances entered as *duty-cycle* will fill standard fields where needed (and keep cycle columns).
"""
)
st.markdown("---")

st.markdown("### Category editor")
st.caption(
    "Define one or more user categories. For each category you can edit: "
    "(i) standard appliances and (ii) duty-cycle appliances. "
    "Time windows are expressed in **minutes** (0–1440)."
)

# Session state structure:
# st.session_state.categories = {
#   "Households": {"num_users": 100, "user_preference": 1, "std_df": ..., "duty_df": ...},
#   ...
# }
if "categories" not in st.session_state:
    st.session_state.categories = {
        "Households": {
            "num_users": 100,
            "user_preference": 1,
            "std_df": _default_std_df(),
            "duty_df": _default_duty_df(),
        }
    }

# Add/remove categories
col_1,col_2 = st.columns([1, 1])
new_name = col_1.text_input("Add new category", placeholder="e.g. Health facility tier 1")
if col_1.button("➕ Add", use_container_width=True):
    name = (new_name or "").strip()
    if not name:
        st.warning("Insert a category name first.")
    elif name in st.session_state.categories:
        st.warning("Category already exists.")
    else:
        st.session_state.categories[name] = {
            "num_users": 1,
            "user_preference": 1,
            "std_df": _default_std_df().iloc[0:0].copy(),   # start empty
            "duty_df": _default_duty_df().iloc[0:0].copy(), # start empty
        }

del_name = col_2.selectbox(
    "Remove category",
    options=["—"] + list(st.session_state.categories.keys()),
    index=0
)
if col_2.button("🗑️ Remove selected", use_container_width=True):
    if del_name != "—":
        st.session_state.categories.pop(del_name, None)
    else:
        st.warning("Select a category to remove.")

st.markdown("---")

# Render each category expander
for cat_name, meta in st.session_state.categories.items():
    with st.expander(f"{cat_name}", expanded=False):
        c1, c2 = st.columns([1, 1])
        meta["num_users"] = int(c1.number_input(
            "Number of users [#]",
            min_value=1,
            value=int(meta.get("num_users", 1)),
            step=1,
            key=f"{cat_name}__num_users"
        ))
        meta["user_preference"] = int(c2.number_input(
            "User preference [-]",
            min_value=1,
            value=int(meta.get("user_preference", 1)),
            step=1,
            key=f"{cat_name}__user_pref"
        ))

        st.markdown("**Standard appliances (no duty cycles)**")
        st.caption(
            "Use this table for appliances that can be represented by windows + total daily time. "
            "Windows are in **minutes** (0–1440). Example: 18:00→1080, 23:59/24:00→1440."
        )

        std_pretty = df_to_pretty(meta["std_df"], STD_COLS)

        edited_std_pretty = st.data_editor(
            std_pretty,
            num_rows="dynamic",
            use_container_width=True,
            key=f"{cat_name}__std_editor",
            column_config=STD_EDITOR_CONFIG,
        )

        edited_std = df_to_canon(edited_std_pretty)

        # protect against paste adding random columns:
        meta["std_df"] = edited_std.reindex(columns=STD_COLS)

        st.markdown("**Duty-cycle appliances (cycle segments visible)**")
        st.caption(
            "Use this table for cyclic appliances (e.g., fridges/freezers) where operation is defined by "
            "one or more cycle templates (power+duration segments + admissible cycle windows)."
        )

        duty_pretty = df_to_pretty(meta["duty_df"], DUTY_COLS)

        edited_duty_pretty = st.data_editor(
            duty_pretty,
            num_rows="dynamic",
            use_container_width=True,
            key=f"{cat_name}__duty_editor",
            column_config=DUTY_EDITOR_CONFIG,
        )

        edited_duty = df_to_canon(edited_duty_pretty)
        meta["duty_df"] = edited_duty.reindex(columns=DUTY_COLS)

        st.markdown("**Time-window visual check (hourly heatmap)**")
        st.caption(
            "Each row is an appliance; colored cells indicate which hours are covered by at least one functioning window."
        )

        # Combine both (for visualization only)
        viz_df = pd.concat(
            [
                meta["std_df"][["name","w1_start_min","w1_end_min","w2_start_min","w2_end_min","w3_start_min","w3_end_min"]],
                meta["duty_df"].reindex(columns=["name","w1_start_min","w1_end_min"]).assign(
                    w2_start_min=0, w2_end_min=0, w3_start_min=0, w3_end_min=0
                )[["name","w1_start_min","w1_end_min","w2_start_min","w2_end_min","w3_start_min","w3_end_min"]],
            ],
            ignore_index=True
        )

        _plot_windows_heatmap(viz_df, title=f"{cat_name} — functioning windows (hourly)")

st.subheader("Build & Download full RAMP Excel")

template_df = load_template_df(TEMPLATE_PATH)

c1, c2 = st.columns([1, 2])
build_clicked = c1.button("🧱 Build full Excel", type="primary", use_container_width=True)

if build_clicked:
    full_df = compile_full_ramp_df(st.session_state.categories, template_df)
    st.session_state["full_ramp_df_preview"] = full_df
    st.success(f"Built full RAMP input with {len(full_df)} appliance rows.")

if "full_ramp_df_preview" in st.session_state:
    full_df = st.session_state["full_ramp_df_preview"]

    st.caption("Preview (first rows of the compiled full-template file):")
    st.dataframe(full_df.head(30), use_container_width=True)

    # write to excel in-memory
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        full_df.to_excel(writer, index=False, sheet_name="Sheet1")
    buf.seek(0)

    st.download_button(
        "⬇️ Download full RAMP input Excel",
        data=buf.getvalue(),
        file_name="ramp_input_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

