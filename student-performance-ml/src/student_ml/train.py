from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict
import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .modeling import build_preprocessor

IDENTIFIER_COLS = {"id", "name", "address"}  # add more if needed


@dataclass
class TrainArtifacts:
    passfail_model_path: str
    metadata_path: str
    passfail_accuracy: float
    passfail_report: str


def _add_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date_Of_Birth" in df.columns and "AL_Exam_Year" in df.columns:
        dob = pd.to_datetime(df["date_Of_Birth"], errors="coerce", dayfirst=True)
        birth_year = dob.dt.year
        exam_year = pd.to_numeric(df["AL_Exam_Year"], errors="coerce")
        df["age_at_exam"] = exam_year - birth_year
    return df


def _normalize_final_status(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip().str.lower()

    label_map = {
        "pass": "Pass", "passed": "Pass", "p": "Pass", "1": "Pass", "1.0": "Pass",
        "true": "Pass", "yes": "Pass", "y": "Pass",
        "fail": "Fail", "failed": "Fail", "f": "Fail", "0": "Fail", "0.0": "Fail",
        "false": "Fail", "no": "Fail", "n": "Fail",
    }
    return raw.map(lambda v: label_map.get(v, v))


def _select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c == target_col:
            continue
        if c.endswith("_Grade"):
            continue
        if c in IDENTIFIER_COLS:
            continue
        cols.append(c)
    return cols


def train_all(
    input_csv: str,
    out_dir: str,
    *,
    target_col: str = "Final_Status",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainArtifacts:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = _add_age_feature(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    y = _normalize_final_status(df[target_col])
    mask = y.isin(["Pass", "Fail"])

    n_labeled = int(mask.sum())
    if n_labeled < 30:
        raise ValueError(
            f"Not enough labeled rows to train. Found {n_labeled} rows with Pass/Fail labels in '{target_col}'. "
            f"Please label at least ~30 students (more is better)."
        )

    df_l = df.loc[mask].copy()
    y_l = y.loc[mask].copy()

    if y_l.nunique() < 2:
        raise ValueError("Only one class found in labeled data (all Pass or all Fail). Need both classes.")

    feature_cols = _select_feature_columns(df_l, target_col=target_col)
    X = df_l[feature_cols].copy()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_l, test_size=test_size, random_state=random_state, stratify=y_l
    )

    pre = build_preprocessor(X, feature_cols)
    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    report = classification_report(yte, pred)
    cm = confusion_matrix(yte, pred, labels=["Fail", "Pass"])

    print("\nClass distribution (labeled rows):")
    print(y_l.value_counts().to_string())
    print("\nAccuracy:", acc)
    print("\nClassification report:\n", report)
    print("\nConfusion matrix (rows=true, cols=pred) [Fail, Pass]:\n", cm)

    model_path = os.path.join(out_dir, "passfail_model.pkl")
    joblib.dump(pipe, model_path)

    metadata: Dict[str, object] = {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "identifier_cols_dropped": sorted(list(IDENTIFIER_COLS)),
        "derived_features": ["age_at_exam"] if "age_at_exam" in df.columns else [],
        "n_labeled_rows_used": int(len(df_l)),
        "random_state": random_state,
        "test_size": test_size,
    }
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return TrainArtifacts(
        passfail_model_path=model_path,
        metadata_path=metadata_path,
        passfail_accuracy=acc,
        passfail_report=report,
    )