# src/student_ml/features.py
from __future__ import annotations
import pandas as pd
import numpy as np

IDENTIFIER_COLS = {
    "id", "name", "address"
}

TARGET_COLS = {
    "Final_Status"
}

def add_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date_Of_Birth" in df.columns and "AL_Exam_Year" in df.columns:
        dob = pd.to_datetime(df["date_Of_Birth"], errors="coerce")
        df["birth_year"] = dob.dt.year
        df["age_at_exam"] = pd.to_numeric(df["AL_Exam_Year"], errors="coerce") - df["birth_year"]
    return df

def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in IDENTIFIER_COLS:
            continue
        if c in TARGET_COLS:
            continue
        # Don't use grade label columns as features (if you have them)
        if c.endswith("_Grade"):
            continue
        cols.append(c)
    return cols