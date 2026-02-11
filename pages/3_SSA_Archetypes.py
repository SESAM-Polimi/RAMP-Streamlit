# pages/2_SSA_Archetypes.py
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from config.path_manager import PM
from core.archetypes_core import demand_calculation


st.title("Sub-Saharan Africa Load Archetypes")
st.caption("Generate a one-year hourly demand profile from pre-defined SSA archetypes.")

st.markdown(
    """
    This page exposes a library of **bottom-up demand archetypes** for
    Sub-Saharan Africa (SSA), derived from typical electricity usage
    patterns for households, schools, and health facilities. These archetypes are the result of a recent study that developed 100 household archetypes, 
    supplemented by 5 health center archetypes and 1 school archetype, characterized by different sets of appliances (wealth parameter), seasonal variations in appliance usage (latitude parameter), 
    and different seasonal use of ambient cooling devices (climate zone parameter). These archetypes reflect typical patterns of electricity usage in off-grid rural areas, representing the collective energy consumption behaviors of the community. Specifically, the study identified:
    - **5 Wealth Tiers**: Each level corresponds to a different basket of appliances, based on a systematic review of literature on electricity use in rural Sub-Saharan Africa.
    - **5 Latitude Zones**: Latitude determines different sunrise and sunset hours, influencing seasonal changes in appliance usage times.
    - **4 Climate Zones**: Different climates across Sub-Saharan Africa affect the need for cooling in households at various times of the year.
    """
)

st.image(PM.assets_dir / "archetypes.png")

st.markdown(
    """
    The methodology is described in:

    > *Archetypes of Rural Users in Sub-Saharan Africa for Load Demand Estimation*  
    > [Archetypes paper – ResearchGate](https://www.researchgate.net/publication/376763546_Archetypes_of_Rural_Users_in_Sub-Saharan_Africa_for_Load_Demand_Estimation)

    The output is a **single synthetic year** at **hourly resolution (8760 hours)**:
    - One **aggregated demand profile** (`Load` column)
    - Optional **per-user profiles** (by household tier, hospital tier, and schools)
"""
)

# -----------------------------------------------------------------------------
# 1. Location and cooling regime
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Location and Cooling regime")
st.markdown(
    """
    **Latitude** influences the **daily and seasonal timing** of electricity use
    (primarily lighting, cooking, and cooling) by modifying sunrise/sunset hours
    across the year. In the archetype study, latitudes across Sub-Saharan Africa
    were grouped into five representative bands, covering roughly **30°S to 20°N**,
    corresponding to where the profiles are valid. Values outside this range may not
    reflect realistic seasonal appliance behaviour and are therefore not supported.

    **Cooling regime** indicates when cooling appliances are used throughout the year:
    - **NC** - No cooling use
    - **AY** - Cooling all year
    - **OM** - Cooling October → March
    - **AS** - Cooling April → September

    These features allow household demand patterns to vary both **seasonally**
    and **behaviourally**, consistent with the climatic and socio-economic diversity
    observed across rural Sub-Saharan African communities.
    """
)


col1, col2 = st.columns(2)
lat = col1.number_input(
    "Latitude [°]",
    min_value=-40.0,
    max_value=30.0,
    value=0.0,
    step=0.5,
    help="Used to select the climatic zone (F1–F5) in the archetype library.",
)
cooling_period = col2.selectbox(
    "Cooling period",
    ["NC", "AY", "OM", "AS"],
    index=0,
    help="Cooling regime used in the archetype profiles (e.g. No Cooling, All Year, etc.).",
)

# -----------------------------------------------------------------------------
# 2. Demand composition
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Demand composition")
st.markdown(
    """
    **Household Wealth Tiers (1 → 5)**  
    Household archetypes differ by **appliance ownership**, **usage intensity**, and
    **time-of-use patterns**. Higher tiers reflect a broader appliance basket
    (e.g. lighting, phone charging, radios, TVs, cooling fans, small ICT), more
    diversified activities, and generally increased evening/night consumption.
    These tiers were constructed from literature on rural electrification and
    observed demand profiles in Sub-Saharan Africa.

    **Health Facilities (5 archetypes)**  
    Represent a spectrum from small rural dispensaries to sub-county hospitals.
    As tier increases, loads become more **daytime-oriented** with higher medical
    equipment and auxiliary service usage (e.g. lighting, refrigeration, ICT),
    consistent with progressively larger operational capacity.
    - Tier 1: Rural dispensary 
    - Tier 2: Basic health center 
    - Tier 3: Medium capacity 
    - Tier 4: District-level 
    - Tier 5: Sub-county hospital

    **School (1 archetype)**  
    Represents a rural primary school with characteristic **daytime loads** and
    peaks around morning and early afternoon activity periods (teaching, ICT,
    lighting, small appliances). No residential or evening loads are assumed.
    """
)

col_hh = st.columns(5)
h1 = col_hh[0].number_input("Households Tier 1", min_value=0.0, value=0.0, step=10.0)
h2 = col_hh[1].number_input("Households Tier 2", min_value=0.0, value=0.0, step=10.0)
h3 = col_hh[2].number_input("Households Tier 3", min_value=0.0, value=0.0, step=10.0)
h4 = col_hh[3].number_input("Households Tier 4", min_value=0.0, value=0.0, step=10.0)
h5 = col_hh[4].number_input("Households Tier 5", min_value=0.0, value=0.0, step=10.0)

col_health = st.columns(6)
sch = col_health[0].number_input("Schools", min_value=0.0, value=0.0, step=1.0)
hp1 = col_health[1].number_input("Hospitals Type 1", min_value=0.0, value=0.0, step=1.0)
hp2 = col_health[2].number_input("Hospitals Type 2", min_value=0.0, value=0.0, step=1.0)
hp3 = col_health[3].number_input("Hospitals Type 3", min_value=0.0, value=0.0, step=1.0)
hp4 = col_health[4].number_input("Hospitals Type 4", min_value=0.0, value=0.0, step=1.0)
hp5 = col_health[5].number_input("Hospitals Type 5", min_value=0.0, value=0.0, step=1.0)

st.caption(
    """
- **Households**: numbers are **absolute counts**; archetype profiles are normalised per 100 households internally.  
- **Hospitals and schools**: numbers are **absolute facilities** (1 = one facility).  

The resulting load is a static one-year profile.  
"""
)

run_btn = st.button("Generate demand using SSA archetypes")

# -----------------------------------------------------------------------------
# 3. Run and preview
# -----------------------------------------------------------------------------
if run_btn:
    try:
        with st.spinner("Generating archetype-based demand..."):
            total_load, users = demand_calculation(
                lat=lat,
                cooling_period=cooling_period,
                num_h_tier1=h1,
                num_h_tier2=h2,
                num_h_tier3=h3,
                num_h_tier4=h4,
                num_h_tier5=h5,
                num_schools=sch,
                num_hospitals1=hp1,
                num_hospitals2=hp2,
                num_hospitals3=hp3,
                num_hospitals4=hp4,
                num_hospitals5=hp5,
            )

        st.success("Archetype-based demand generated successfully.")

        # Store in session for visualization + downloads
        st.session_state.ssa_total_load = total_load
        st.session_state.ssa_users = users

    except Exception as e:
        st.error(f"Error while generating demand from archetypes: {e}")

# -----------------------------------------------------------------------------
# 4. Visualization – similar UX to RAMP page
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Visualize SSA archetype profiles")
st.markdown("**Daily average load curve (hourly resolution)**")

if "ssa_total_load" not in st.session_state:
    st.info("Generate a profile above to enable visualization.")
else:
    total_load: pd.DataFrame = st.session_state.ssa_total_load
    users = st.session_state.ssa_users or []

    # Build options: Aggregated + per-user
    names = ["Aggregated"]
    names.extend([u.name for u in users])

    selected = st.selectbox("Select profile to visualize", names, index=0)

    # --- Extract selected time series (8760 values expected) ---
    if selected == "Aggregated":
        series = total_load["Load"].to_numpy().ravel()
        title_base = "Aggregated SSA load profile"
    else:
        sel = next((u for u in users if u.name == selected), None)
        if sel is None:
            st.warning(f"Could not find user '{selected}'.")
            st.stop()
        if "Load" in sel.demand_data.columns:
            series = sel.demand_data["Load"].to_numpy().ravel()
        else:
            series = sel.demand_data.to_numpy().ravel()
        title_base = f"SSA archetype — {selected}"

    hours_per_day = 24
    series = np.asarray(series)
    if series.size % hours_per_day != 0:
        st.error(f"Profile length {series.size} is not a multiple of 24.")
        st.stop()

    n_days = series.size // hours_per_day
    mat = series.reshape((n_days, hours_per_day))  # (days, 24)

    # Compute min, max, average across days (for each hour of day)
    prof_min = mat.min(axis=0)
    prof_max = mat.max(axis=0)
    prof_avg = mat.mean(axis=0)

    factor = 1000.0  # display in kW assuming input is in W (adjust if needed)
    y_label = "Power [kW]"

    # --- Plot 1: average daily load + variability band + daily "cloud" ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # cloud of daily profiles
    for day in mat:
        ax.plot(day / factor, linewidth=0.4, alpha=0.25, color="gray")

    # min–max band
    ax.fill_between(
        range(hours_per_day),
        prof_min / factor,
        prof_max / factor,
        alpha=0.25,
        color="lightgray",
        label="Variability range",
    )

    # average
    ax.plot(
        prof_avg / factor,
        linewidth=2.0,
        color="red",
        label="Average daily profile",
    )

    ax.set_title(f"{title_base} — daily average (hourly)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(hours_per_day))
    ax.set_xticklabels([f"{h}:00" for h in range(hours_per_day)], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    st.pyplot(fig)

    # --- Plot 2: inspect a specific day ---
    st.markdown("**Inspect a specific day**")
    day_idx = st.slider(
        "Day of year",
        min_value=1,
        max_value=n_days,
        value=1,
        step=1,
    )
    day_vec = mat[day_idx - 1, :]  # 1-based → 0-based

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(day_vec / factor, linewidth=1.8)
    ax2.set_title(f"Day #{day_idx} — {title_base}")
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel(y_label)
    ax2.set_xticks(range(hours_per_day))
    ax2.set_xticklabels([f"{h}:00" for h in range(hours_per_day)], rotation=45)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# -----------------------------------------------------------------------------
# 5. Downloads – always at the end, based on session state
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Download SSA archetype demand profiles")

if "ssa_total_load" not in st.session_state:
    st.info("No SSA profile available yet. Generate a profile above to enable downloads.")
else:
    total_load: pd.DataFrame = st.session_state.ssa_total_load
    users = st.session_state.ssa_users or []

    st.markdown(
        """
        Export the **one-year hourly profiles** for integration in other tools
        (e.g. MicroGridsPy, OSeMOSYS, PyPSA, Autarky):

        - **Aggregated hourly profile**: 8760 values (`Load` column)  
        - **Per-user hourly profiles**: one CSV per user (Household tiers, Hospital tiers, School)
"""
    )

    # 1) Aggregated demand CSV (8760 × 1, column 'Load')
    agg_csv = total_load.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download aggregated SSA demand (hourly, 8760 values)",
        data=agg_csv,
        file_name="ssa_archetypes_aggregated_hourly.csv",
        mime="text/csv",
    )

    # 2) Per-user profiles as ZIP
    if users:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for u in users:
                buf = io.StringIO()
                u.demand_data.to_csv(buf, index=False)
                zf.writestr(f"{u.name}_hourly.csv", buf.getvalue())
        zip_buf.seek(0)
        st.download_button(
            "Download all user profiles (hourly, ZIP)",
            data=zip_buf,
            file_name="ssa_archetypes_users_hourly.zip",
            mime="application/zip",
        )
