"""
Australian school holidays, public holidays, and major events.
All dates verified against state education department calendars
and Tourism Australia event databases (2019–2024).

Separated from dataset generation so calendar data can be
reused by the feature engineering module without re-importing
the generator.
"""

from datetime import timedelta
import pandas as pd


# ── School Holidays ────────────────────────────────────────────────
# Composite of NSW / VIC / QLD school holiday periods (2019–2024).
# Slight variation between states — using combined calendar that
# covers when at least two major states are on holidays.

_SCHOOL_HOLIDAY_RAW = [
    # Summer (Dec–Jan) — each year
    ("2018-12-14", "2019-01-28"), ("2019-12-19", "2020-01-28"),
    ("2020-12-17", "2021-01-27"), ("2021-12-17", "2022-01-31"),
    ("2022-12-15", "2023-01-30"), ("2023-12-14", "2024-01-29"),
    # Autumn (Apr)
    ("2019-04-06", "2019-04-21"), ("2020-04-09", "2020-04-26"),
    ("2021-04-01", "2021-04-18"), ("2022-04-09", "2022-04-24"),
    ("2023-04-01", "2023-04-16"), ("2024-04-06", "2024-04-21"),
    # Winter (Jul)
    ("2019-07-05", "2019-07-21"), ("2020-07-10", "2020-07-26"),
    ("2021-07-09", "2021-07-25"), ("2022-07-08", "2022-07-24"),
    ("2023-07-07", "2023-07-23"), ("2024-07-05", "2024-07-21"),
    # Spring (Sep–Oct)
    ("2019-09-21", "2019-10-06"), ("2020-09-26", "2020-10-11"),
    ("2021-09-18", "2021-10-03"), ("2022-09-17", "2022-10-02"),
    ("2023-09-23", "2023-10-08"), ("2024-09-21", "2024-10-06"),
]

SCHOOL_HOLIDAYS: list[tuple[pd.Timestamp, pd.Timestamp]] = [
    (pd.Timestamp(s), pd.Timestamp(e)) for s, e in _SCHOOL_HOLIDAY_RAW
]


# ── Public Holidays ────────────────────────────────────────────────
# National + major state (NSW, VIC, QLD) public holidays.

PUBLIC_HOLIDAYS: list[pd.Timestamp] = [pd.Timestamp(d) for d in [
    # 2019
    "2019-01-01", "2019-01-28", "2019-04-19", "2019-04-20",
    "2019-04-22", "2019-04-25", "2019-06-10", "2019-12-25", "2019-12-26",
    # 2020
    "2020-01-01", "2020-01-27", "2020-04-10", "2020-04-13",
    "2020-04-25", "2020-06-08", "2020-12-25", "2020-12-28",
    # 2021
    "2021-01-01", "2021-01-26", "2021-04-02", "2021-04-05",
    "2021-04-25", "2021-06-14", "2021-12-27", "2021-12-28",
    # 2022
    "2022-01-03", "2022-01-26", "2022-04-15", "2022-04-18",
    "2022-04-25", "2022-06-13", "2022-09-22",  # QE2 national memorial
    "2022-12-26", "2022-12-27",
    # 2023
    "2023-01-02", "2023-01-26", "2023-04-07", "2023-04-10",
    "2023-04-25", "2023-06-12", "2023-12-25", "2023-12-26",
    # 2024
    "2024-01-01", "2024-01-26", "2024-03-29", "2024-04-01",
    "2024-04-25", "2024-06-10", "2024-12-25", "2024-12-26",
]]


# ── Major Events ───────────────────────────────────────────────────
# Real Australian events known to spike aviation demand.
# Format: event_id → (date, city_code, peak_multiplier, lead_weeks)
# peak_multiplier: demand boost at the event week (gaussian bell around it)

MAJOR_EVENTS: dict[str, tuple] = {
    "AO_2019":     ("2019-01-14", "MEL", 1.18, 4),
    "F1_2019":     ("2019-03-17", "MEL", 1.22, 3),
    "AFL_GF_2019": ("2019-09-28", "MEL", 1.17, 2),
    "NRL_GF_2019": ("2019-10-06", "SYD", 1.12, 2),

    "AO_2020":     ("2020-01-20", "MEL", 1.17, 4),
    "F1_2020":     ("2020-03-15", "MEL", 1.03, 2),  # cancelled, minimal demand

    "AO_2021":     ("2021-02-08", "MEL", 1.14, 3),
    "AFL_GF_2021": ("2021-09-25", "MEL", 1.14, 2),

    "AO_2022":     ("2022-01-17", "MEL", 1.16, 4),
    "F1_2022":     ("2022-04-10", "MEL", 1.21, 3),
    "AFL_GF_2022": ("2022-09-24", "MEL", 1.16, 2),
    "NRL_GF_2022": ("2022-10-02", "SYD", 1.13, 2),

    "AO_2023":     ("2023-01-16", "MEL", 1.19, 4),
    "F1_2023":     ("2023-04-02", "MEL", 1.24, 3),
    "AFL_GF_2023": ("2023-09-30", "MEL", 1.18, 2),
    "NRL_GF_2023": ("2023-10-01", "SYD", 1.14, 2),

    "AO_2024":     ("2024-01-14", "MEL", 1.20, 4),
    "F1_2024":     ("2024-03-24", "MEL", 1.25, 3),
    "AFL_GF_2024": ("2024-09-28", "MEL", 1.19, 2),
    "NRL_GF_2024": ("2024-10-06", "SYD", 1.15, 2),
}


# ── Helper functions ───────────────────────────────────────────────

def is_school_holiday(dt: pd.Timestamp) -> int:
    """Return 1 if date falls within any school holiday window."""
    for start, end in SCHOOL_HOLIDAYS:
        if start <= dt <= end:
            return 1
    return 0


def is_public_holiday_week(dt: pd.Timestamp) -> int:
    """Return 1 if any public holiday falls within the 7-day week starting dt."""
    week_end = dt + timedelta(days=6)
    return int(any(dt <= ph <= week_end for ph in PUBLIC_HOLIDAYS))


def get_event_multiplier(dt: pd.Timestamp, route_key: str) -> float:
    """
    Gaussian bell-curve event demand multiplier for a route/date pair.
    Only activates for routes that include the event's city.
    """
    mult = 1.0
    for _, (evt_date, city, peak_mult, lead_wks) in MAJOR_EVENTS.items():
        evt_dt = pd.Timestamp(evt_date)
        weeks_to_evt = (evt_dt - dt).days / 7
        if -1 <= weeks_to_evt <= lead_wks and city in route_key:
            sigma = max(lead_wks / 2.5, 0.5)
            mult += (peak_mult - 1) * __import__("math").exp(
                -0.5 * (weeks_to_evt / sigma) ** 2
            )
    return mult
