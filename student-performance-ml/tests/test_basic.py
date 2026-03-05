import pandas as pd
from student_ml.preprocess import prepare_dataset

def test_prepare_dataset_runs():
    df = pd.DataFrame({
        "Physics": [80, 30],
        "Physics_Selected": ["Selected", "Selected"],
        "Physics_Grade": ["A", "W"]
    })
    prepared = prepare_dataset(df)
    assert "Physics_Selected_bool" in prepared.X.columns
    assert prepared.y_grades is not None
