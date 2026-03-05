from __future__ import annotations

import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .preprocess import prepare_dataset


def predict_all(
    input_csv: str,
    models_dir: str,
    *,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    meta_path = os.path.join(models_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    prepared = prepare_dataset(
        df,
        selected_marker=meta["selected_marker"],
        grade_suffix=meta["grade_suffix"],
        passfail_target_col=meta["passfail_target_col"],
    )

    X = prepared.X
    feature_cols = meta["feature_cols"]
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]

    pf_path = os.path.join(models_dir, "passfail_model.pkl")
    if os.path.exists(pf_path):
        pf = joblib.load(pf_path)
        df["Predicted_Status"] = pf.predict(X)

    g_path = os.path.join(models_dir, "grades_model.pkl")
    if os.path.exists(g_path):
        gm = joblib.load(g_path)
        preds = gm.predict(X)
        grade_cols = meta.get("grade_cols", [])
        n_out = preds.shape[1] if hasattr(preds, "shape") else len(preds[0])
        cols = grade_cols[:n_out]
        for i, col in enumerate(cols):
            df[f"Predicted_{col}"] = preds[:, i]

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df
