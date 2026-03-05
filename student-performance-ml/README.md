# Student Performance ML

Complete working template to train and use ML models to predict:
1) Pass/Fail (Final_Status)
2) Subject-wise grades (Physics_Grade, Chemistry_Grade, ...)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train
```bash
python scripts/train.py --input data/raw/student_data.csv --out models
```

### Predict
```bash
python scripts/predict.py --input data/raw/student_data.csv --models models --out data/predictions/predictions.csv
```

## Labels
- If `Final_Status` exists (Pass/Fail or 1/0), it will be used.
- If `Final_Status` is missing but real grade columns exist, pass/fail is derived:
  If any selected subject has grade `W` -> `Fail`, else `Pass`.

## Selected subjects
If your CSV contains `<Subject>_Selected` columns with values like Selected/Yes/1/True,
the pipeline uses them. Otherwise it assumes a subject is selected if the score is present.
