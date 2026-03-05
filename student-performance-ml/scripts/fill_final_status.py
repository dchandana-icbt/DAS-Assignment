import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib

def normalize_status(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    label_map = {
        "pass": "Pass", "passed": "Pass", "p": "Pass", "1": "Pass", "1.0": "Pass",
        "true": "Pass", "yes": "Pass", "y": "Pass",
        "fail": "Fail", "failed": "Fail", "f": "Fail", "0": "Fail", "0.0": "Fail",
        "false": "Fail", "no": "Fail", "n": "Fail",
    }
    return raw.map(lambda v: label_map.get(v, v))

def add_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date_Of_Birth" in df.columns and "AL_Exam_Year" in df.columns:
        dob = pd.to_datetime(df["date_Of_Birth"], errors="coerce", dayfirst=True)
        df["age_at_exam"] = pd.to_numeric(df["AL_Exam_Year"], errors="coerce") - dob.dt.year
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-col", default="Final_Status")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = add_age_feature(df)

    meta_path = os.path.join(args.models, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feature_cols].copy()

    model_path = os.path.join(args.models, "passfail_model.pkl")
    model = joblib.load(model_path)

    if args.target_col not in df.columns:
        df[args.target_col] = np.nan

    y_norm = normalize_status(df[args.target_col])
    missing_mask = ~y_norm.isin(["Pass", "Fail"])

    preds = model.predict(X.loc[missing_mask])
    df.loc[missing_mask, args.target_col] = preds

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print("Updated file saved to:", args.out)
    print("Rows filled:", int(missing_mask.sum()))

if __name__ == "__main__":
    main()