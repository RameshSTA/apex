"""src.data — data generation, calendar, and feature engineering."""
from .generator import DatasetGenerator
from .features import FeatureEngineer
from .calendar import (
    SCHOOL_HOLIDAYS, PUBLIC_HOLIDAYS, MAJOR_EVENTS,
    is_school_holiday, is_public_holiday_week, get_event_multiplier,
)
