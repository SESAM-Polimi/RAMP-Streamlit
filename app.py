import streamlit as st
from config.path_manager import PM

st.set_page_config(page_title="RAMP Demand Simulator", layout="wide")

st.title("RAMP - Bottom-up Demand Simulation")
st.caption("Build RAMP inputs, run stochastic simulations, and export demand profiles for energy system models.")

# --- About RAMP --------------------------------------------------------------
st.subheader("About RAMP")
st.markdown(
    """
[RAMP](https://rampdemand.org/) is an open-source framework for generating
**high-resolution, stochastic demand profiles** from appliance-level assumptions,
particularly useful when only limited information on user behaviour is available.

This Streamlit app wraps the RAMP engine in a streamlined interface to help you:
- prepare **RAMP-compatible inputs** more easily,
- run **minute-resolution stochastic simulations**, and
- export **aggregated / per-category time series** for planning, dispatch, and optimisation workflows.
"""
)

st.image(PM.assets_dir / "ramp.png", use_container_width=False)

st.markdown(
    """
📖 **RAMP docs:** https://rampdemand.readthedocs.io/en/latest  
💻 **RAMP GitHub:** https://github.com/RAMP-project/RAMP
"""
)

st.markdown("---")

# --- App organisation --------------------------------------------------------
st.subheader("How the app is organised")

st.markdown(
    """
The application is organised into **three pages**, accessible from the Streamlit **sidebar**.
Depending on your workflow, you can either (A) build the Excel inputs directly in the app
or (B) upload your own full RAMP Excel input file and run the simulation.
"""
)

# --- Page 1 ------------------------------------------------------------------
st.markdown("### 1) Inputs Editor")
st.markdown(
    """
Use this page to **create full RAMP Excel inputs directly in the app** (recommended if you want
to avoid editing the full template manually).

Main features:
- Create multiple **user categories** (e.g., Households, Clinic Tier 1, Schools).
- Add **standard appliances** (windows + total daily use time).
- Add **duty-cycle appliances** (cycle segments and admissible cycle windows).
- Quick **visual validation** with a time-window **heatmap** (hourly view).
- Export a **ready-to-run full RAMP Excel** file (with user-friendly / “pretty” headers, internally mapped to canonical fields).
"""
)

# --- Page 2 ------------------------------------------------------------------
st.markdown("### 2) RAMP Simulation")
st.markdown(
    """
Use this page to **run the stochastic demand simulation** with the RAMP engine.

Workflow:
1. Configure the **year structure**:
   - number of seasons (1–4),
   - month allocation per season,
   - optional **week-classes** (e.g., Weekday/Weekend).
2. Upload the **full RAMP Excel input file**:
   - single-archetype mode → upload one full Excel file,
   - multi-archetype mode → upload one full Excel file per archetype (**Season × Week-class**, or Season only).
3. Run the simulation to generate:
   - **minute-resolution daily profiles** (stochastic pool per archetype),
   - a synthetic yearly profile assembled into a **365-day** timeline.
4. Visualise:
   - average daily profiles,
   - min–max variability bands,
   - day-by-day inspection and filtering by season / week-class (when enabled).
5. Export results:
   - per-category minute profiles,
   - aggregated minute profile,
   - aggregated **hourly** profile (recommended for optimisation models).
"""
)

# --- Page 3 ------------------------------------------------------------------
st.markdown("### 3) SSA Archetypes")
st.markdown(
    """
Use this page when you need a **fast, bottom-up demand profile** based on pre-defined
**Sub-Saharan Africa archetypes** (without building full RAMP appliance sheets).

Main features:
- Select climatic context (e.g., by latitude / cooling regime logic).
- Specify number of users by tier (households, schools, health facilities).
- Generate a **one-year hourly aggregated profile (8760)** and optional per-archetype outputs.
- Export CSVs ready for planning and optimisation tools.
"""
)

st.markdown("---")
st.info("👉 Use the **sidebar navigation** to start: typically **Inputs Editor → RAMP Simulation → Download results**.")
