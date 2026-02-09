# core/archetypes_core.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from config.path_manager import PM

# Fixed: one year, hourly resolution
PERIODS_PER_YEAR = 8760


@dataclass
class User:
    name: str
    demand_data: pd.DataFrame


class Household:
    def __init__(self, zone: str, wealth: int, cooling: str, number: float):
        self.zone = zone
        self.wealth = wealth
        self.cooling = cooling
        self.number = number

    def load_demand(self, h_load: pd.DataFrame) -> pd.DataFrame:
        # Household profiles are defined per 100 households in the library
        return self.number / 100.0 * h_load


class Hospital:
    def __init__(self, tier: int, number: float):
        self.tier = tier
        self.number = number

    def load_demand(self, h_load: pd.DataFrame) -> pd.DataFrame:
        return self.number * h_load


class School:
    def __init__(self, number: float):
        self.number = number

    def load_demand(self, s_load: pd.DataFrame) -> pd.DataFrame:
        return self.number * s_load


def determine_zone(lat: float) -> str:
    """Map latitude to archetype zone; valid only for Sub-Saharan Africa."""
    if 10 <= lat <= 20:
        return "F1"
    elif -10 <= lat < 10:
        return "F2"
    elif -20 <= lat < -10:
        return "F3"
    elif -30 <= lat < -20:
        return "F4"
    elif lat < -30:
        return "F5"
    else:
        raise ValueError(
            "Latitude out of range. Archetypes are valid only for Sub-Saharan "
            "Africa (lat between -30° and 20°)."
        )


def aggregate_load(load_data: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Aggregate raw series to a given number of periods per year.

    The archetype library is typically hourly already (8760 points). If the
    length differs, this groups by blocks to match the requested `periods`.
    """
    total_steps = len(load_data)
    if total_steps == periods:
        return load_data

    agg_factor = total_steps // periods
    if agg_factor <= 0:
        raise ValueError(
            f"Cannot aggregate from {total_steps} steps to {periods} periods."
        )
    aggregated = load_data.groupby(load_data.index // agg_factor).sum()

    # If we overshoot slightly due to integer division, trim
    if len(aggregated) > periods:
        aggregated = aggregated.iloc[:periods, :]

    return aggregated


def _archetypes_folder() -> str:
    # Folder with the archetype Excel files
    # e.g. config/assets/archetypes_library
    return str(PM.config_dir / "archetypes_library")


def load_household_data(household: Household) -> pd.DataFrame:
    h_load_name = f"{household.cooling}_{household.zone}_Tier-{household.wealth}"
    file_path = os.path.join(_archetypes_folder(), f"{h_load_name}.xlsx")
    h_load = pd.read_excel(file_path, skiprows=0, usecols="B")
    return aggregate_load(h_load, PERIODS_PER_YEAR)


def load_hospital_data(hospital: Hospital) -> pd.DataFrame:
    hospital_load_name = f"HOSPITAL_Tier-{hospital.tier}"
    file_path = os.path.join(_archetypes_folder(), f"{hospital_load_name}.xlsx")
    h_load = pd.read_excel(file_path, skiprows=0, usecols="B")
    return aggregate_load(h_load, PERIODS_PER_YEAR)


def load_school_data(school: School) -> pd.DataFrame:
    school_load_name = "SCHOOL"
    file_path = os.path.join(_archetypes_folder(), f"{school_load_name}.xlsx")
    s_load = pd.read_excel(file_path, skiprows=0, usecols="B")
    return aggregate_load(s_load, PERIODS_PER_YEAR)


def demand_calculation(
    lat: float,
    cooling_period: str,
    num_h_tier1: float,
    num_h_tier2: float,
    num_h_tier3: float,
    num_h_tier4: float,
    num_h_tier5: float,
    num_schools: float,
    num_hospitals1: float,
    num_hospitals2: float,
    num_hospitals3: float,
    num_hospitals4: float,
    num_hospitals5: float,
) -> Tuple[pd.DataFrame, List[User]]:
    """
    Build a single-year hourly demand profile from SSA archetypes.

    Returns
    -------
    total_load : DataFrame
        Shape (8760, 1), column 'Load' in W (or whatever unit in the library).
    users : list[User]
        One User per non-zero group (tier / facility type), each with
        demand_data of shape (8760, 1).
    """
    if lat is None:
        raise ValueError("Latitude is not set. Please provide a latitude in °.")

    periods = PERIODS_PER_YEAR
    zone = determine_zone(lat)
    users: List[User] = []

    # ---------------- Households ----------------
    households = [
        Household(zone, 1, cooling_period, num_h_tier1),
        Household(zone, 2, cooling_period, num_h_tier2),
        Household(zone, 3, cooling_period, num_h_tier3),
        Household(zone, 4, cooling_period, num_h_tier4),
        Household(zone, 5, cooling_period, num_h_tier5),
    ]

    load_households = pd.DataFrame(0.0, index=range(periods), columns=["Load"])
    for tier, hh in enumerate(households, start=1):
        if hh.number > 0:
            base = hh.load_demand(load_household_data(hh))
            load_households["Load"] += base.iloc[:, 0]

            # Per-household-tier profile
            users.append(
                User(
                    name=f"Household_Tier_{tier}",
                    demand_data=base.rename(columns={base.columns[0]: "Load"}),
                )
            )

    # ---------------- Hospitals ----------------
    hospitals = [
        Hospital(1, num_hospitals1),
        Hospital(2, num_hospitals2),
        Hospital(3, num_hospitals3),
        Hospital(4, num_hospitals4),
        Hospital(5, num_hospitals5),
    ]

    load_hosp = pd.DataFrame(0.0, index=range(periods), columns=["Load"])
    for tier, hosp in enumerate(hospitals, start=1):
        if hosp.number > 0:
            base = hosp.load_demand(load_hospital_data(hosp))
            load_hosp["Load"] += base.iloc[:, 0]

            users.append(
                User(
                    name=f"Hospital_Tier_{tier}",
                    demand_data=base.rename(columns={base.columns[0]: "Load"}),
                )
            )

    # ---------------- Schools ----------------
    load_school = pd.DataFrame(0.0, index=range(periods), columns=["Load"])
    if num_schools > 0:
        school = School(num_schools)
        base = school.load_demand(load_school_data(school))
        load_school["Load"] += base.iloc[:, 0]

        users.append(
            User(
                name="School",
                demand_data=base.rename(columns={base.columns[0]: "Load"}),
            )
        )

    # ---------------- Aggregate ----------------
    load_total = load_households + load_hosp + load_school
    if load_total["Load"].sum() == 0:
        raise ValueError(
            "Total load is zero. Provide at least one non-zero demand "
            "(households, hospitals, or schools)."
        )

    load_total.columns = ["Load"]
    return load_total, users
