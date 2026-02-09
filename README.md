# **RAMP Load Demand Simulation – Streamlit Application**

<p align="center">
  <img src="config/assets/ramp.png" width="700" alt="RAMP for Bottom-Up Load Demand Simulation">
</p>

This is a **Streamlit-based** interface that makes the **bottom-up, appliance-level load demand modelling** capabilities of the RAMP framework accessible to a broader audience. RAMP generates realistic high-resolution electricity demand time series by simulating user behaviour, appliance operation, and stochastic variability at minute-scale resolution. Traditionally, this workflow requires detailed spreadsheets and Python scripting; here, the same pipeline is exposed through an interactive graphical interface.

The simulator guides the user through defining a **temporal representation of the year**, assembling one or more **behavioural archetypes** (season × week-class), running the stochastic RAMP engine, and analysing the final synthetic 365-day demand profiles. Visualisation tools allow inspection of average daily shapes, variability envelopes, and specific instances of the load realisation, with export options suitable for energy planning, techno-economic analysis, feasibility studies, and optimisation models.

In addition to the RAMP modelling workflow, the app includes a complementary pathway based on **Sub-Saharan Africa (SSA) demand archetypes**. This mode generates load profiles from a library of empirically derived user archetypes for households, schools, and health facilities, differentiated by climatic zone, socio-economic tier, and usage characteristics. This enables rapid scenario construction in data-scarce contexts where demographic or service information is available, but detailed appliance breakdowns are not.

Both pathways share a common design objective: to provide transparent, reproducible, and well-structured demand profiles ready for downstream energy system modelling.

---

# **Application Structure**

The app consists of two main sections, selectable from the sidebar.

## **1. RAMP Simulation**

This section exposes the full RAMP stochastic workflow to the user:

### **Year Structure & Temporal Modelling**
Users define how the year is partitioned in terms of:
- **Seasons** (1–4), with arbitrary month assignments
- Optional **within-week differentiation**, such as:
  - *Weekday vs Weekend*
  - Custom week-classes (up to 7)

Each *(season × week-class)* combination becomes a **day archetype**, i.e. a stochastic pool of daily profiles that represent typical usage patterns for that time period.

### **Input Preparation & Archetype Building**
RAMP requires full appliance-level Excel inputs. The app allows users to upload **simplified RAMP Excel files**, one per archetype or a single file for the whole year. These are automatically expanded into full RAMP-compatible inputs and persisted for later reuse.

Two execution modes are supported:

- **Single-archetype mode**: one input governs the whole year; the simulator generates `num_days` synthetic daily profiles and tiles them into a full year.
- **Multi-archetype mode**: each archetype has its own stochastic pool; the synthetic year is assembled by walking the calendar and sampling from the appropriate pool.

This captures:
- stochastic appliance behaviour (intra-archetype variability)
- seasonal & weekly patterns (inter-archetype diversity)

### **Visualisation & Analysis**
The output is a 365×1440 minute profile (or per-user equivalent). Users can:
- visualise **aggregated** and **per-category** profiles
- inspect **daily average curves** with min–max envelopes
- explore **stochastic variability** via daily clouds
- view **specific day instances**
- export **hourly aggregates** for planning tools

## **2. SSA Load Archetypes**

This section provides a fast scenario alternative based on empirically derived SSA archetypes. Rather than appliance inventories, users specify the **composition** of the settlement:

- households by **tier (1–5)**
- **schools**
- **hospitals** (5 types)

Latitude selects one of the climatic zones (F1–F5), and a **cooling regime** reflects different household behaviours. The underlying library is based on:

> *Archetypes of Rural Users in Sub-Saharan Africa for Load Demand Estimation*  
> https://www.researchgate.net/publication/376763546_Archetypes_of_Rural_Users_in_Sub-Saharan_Africa_for_Load_Demand_Estimation

The methodology aggregates hourly profiles derived from surveys, measurement campaigns, and behavioural modelling — suitable for electrification planning, mini-grid simulations, and access planning under SDG7 frameworks.

<p align="center">
  <img src="config/assets/archetypes.png" width="700" alt="RAMP for Bottom-Up Load Demand Simulation">
</p>

This mode outputs:
- a single **aggregated hourly profile**
- **per-user category profiles** (households by tier, schools, hospitals)

It prioritises accessibility, speed, and scenario logic for data-scarce environments.

---

# **Inputs & Outputs**

### **Inputs**
- Simplified RAMP Excel templates (appliance-level)
- SSA archetype composition (tiers, schools, hospitals)

### **Outputs**
All outputs are written in **CSV** format:

| File | Description |
|---|---|
| `profile_aggregated.csv` | 365×1440 aggregated RAMP profile |
| `profile_aggregated_hourly.csv` | 8760 hourly profile (RAMP) |
| `ssa_aggregated_hourly.csv` | hourly aggregated SSA archetypes |
| `profile_<user>.csv` | per-user appliance-level profiles (RAMP) |
| `ssa_<user>.csv` | per-user archetype profiles (SSA) |

These profiles are directly usable in techno-economic optimisation, dispatch models, power flow studies, and feasibility assessments.

---

# **Installation**

### **Conda (recommended)**

```bash
conda env create -f environment.yml
conda activate ramp_streamlit
```

### **Running**

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

# **Contact**

**Alessandro Onori**  
📧 alessandro.onori@polimi.it  

Technical Advisors  
- Riccardo Mereu - Politecnico di Milano  
- Emanuela Colombo - Politecnico di Milano

---

# **License**

European Union Public Licence (EUPL v1.1).
