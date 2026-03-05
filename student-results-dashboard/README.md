# Student Results Streamlit Dashboard

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` - main Streamlit dashboard
- `data/student_data_with_final_status.csv` - attached model output used as default dataset

## Notes
- The dashboard includes:
  - 12+ normal/statistical visuals
  - 5 spatial visuals
  - 5+ network visuals
- The source dataset contains **locality labels** (`address`) but not true latitude/longitude.
  Therefore, the geographic section uses **deterministic schematic coordinates** derived from the address field.
  These maps are appropriate for exploratory storytelling, clustering, density, and comparative pattern analysis,
  but they should not be interpreted as exact real-world geocoded positions.
