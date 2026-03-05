from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GRADES


def normalize_selected(value) -> bool:
    """Normalize Selected column values like 'Selected', 'Yes', 1, True."""
    if value is None:
        return False
    if isinstance(value, (int, float)) and not np.isnan(value):
        try:
            return bool(int(value))
        except Exception:
            return False
    s = str(value).strip().lower()
    return s in {"selected", "yes", "y", "true", "1"}


def clean_grade_series(s: pd.Series) -> pd.Series:
    """Normalize grades to A/B/C/S/W and set others to NaN."""
    out = s.astype(str).str.strip().str.upper()
    out = out.where(out.isin(GRADES), np.nan)
    return out
