"""Target construction, leakage guard, and temporal split."""

from __future__ import annotations

import pandas as pd

ID_COLUMNS = ["encounter_id", "patient_nbr"]
LEAKY_COLUMNS = ["discharge_disposition_id"]
TARGET_COLUMN = "readmitted"


def make_target(df: pd.DataFrame) -> pd.Series:
    """1 when the patient was readmitted within 30 days."""
    return (df[TARGET_COLUMN] == "<30").astype(int)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop identifiers, the target, and columns that leak the outcome."""
    drop = [c for c in ID_COLUMNS + LEAKY_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    return df.drop(columns=drop), make_target(df)


def temporal_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by encounter_id order as a proxy for time.

    encounter_id is sequential but not a real timestamp. This split prevents
    random-split optimism; a true date-based split is future work.
    """
    ordered = df.sort_values("encounter_id").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
