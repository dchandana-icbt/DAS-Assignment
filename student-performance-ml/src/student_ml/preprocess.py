from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import SUBJECTS
from .utils import normalize_selected, clean_grade_series


@dataclass
class PreparedData:
    X: pd.DataFrame
    feature_cols: List[str]
    subjects_present: List[str]
    selected_marker: str
    grade_suffix: str
    passfail_target_col: str
    grade_cols: List[str]
    y_passfail: Optional[pd.Series]
    y_grades: Optional[pd.DataFrame]


def prepare_dataset(
    df: pd.DataFrame,
    *,
    selected_marker: str = "Selected",
    grade_suffix: str = "_Grade",
    passfail_target_col: str = "Final_Status",
) -> PreparedData:
    subjects_present = [s for s in SUBJECTS if s in df.columns]

    for s in subjects_present:
        df[s] = pd.to_numeric(df[s], errors="coerce")

    selected_map: Dict[str, pd.Series] = {}
    for s in subjects_present:
        sel_col = f"{s}_{selected_marker}"
        if sel_col in df.columns:
            selected_map[s] = df[sel_col].apply(normalize_selected).astype(bool)
        else:
            selected_map[s] = df[s].notna()

    X = df.copy()
    for s in subjects_present:
        X[s] = X[s].where(selected_map[s], np.nan)
        X[f"{s}_{selected_marker}_bool"] = selected_map[s].astype(int)

    y_passfail = None
    if passfail_target_col in df.columns:
        y_passfail = df[passfail_target_col].astype(str).str.strip().str.title()
        y_passfail = y_passfail.replace({"1": "Pass", "0": "Fail"})

    grade_cols = [f"{s}{grade_suffix}" for s in subjects_present if f"{s}{grade_suffix}" in df.columns]
    y_grades = None
    if grade_cols:
        y_grades = pd.DataFrame({c: clean_grade_series(df[c]) for c in grade_cols})

    if y_passfail is None and y_grades is not None and not y_grades.empty:
        def derive_pf(i: int) -> str:
            for s in subjects_present:
                if not selected_map[s].iloc[i]:
                    continue
                col = f"{s}{grade_suffix}"
                if col in y_grades.columns and y_grades[col].iloc[i] == "W":
                    return "Fail"
            return "Pass"
        y_passfail = pd.Series([derive_pf(i) for i in range(len(df))], index=df.index, name=passfail_target_col)

    target_cols = set([passfail_target_col] + grade_cols)
    feature_cols = [c for c in X.columns if c not in target_cols]
    X = X[feature_cols]

    return PreparedData(
        X=X,
        feature_cols=feature_cols,
        subjects_present=subjects_present,
        selected_marker=selected_marker,
        grade_suffix=grade_suffix,
        passfail_target_col=passfail_target_col,
        grade_cols=grade_cols,
        y_passfail=y_passfail,
        y_grades=y_grades,
    )
