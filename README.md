# Real Estate Investment Advisor

A complete starter project based on the supplied project brief: preprocessing, 20 EDA questions, feature engineering, classification, regression, Streamlit, and MLflow.

## Project structure

- `data/india_housing_prices.csv` — original dataset
- `01_data_preprocessing_and_eda.ipynb` — data audit, cleaning, feature engineering, EDA charts and exports
- `src/train_models.py` — classification and current-price regression model training, evaluation, and optional MLflow logging
- `app.py` — Streamlit interactive prototype
- `outputs/` — created by notebook/scripts (cleaned data, charts, tables, metrics)
- `models/` — trained model artifacts

## Setup (VS Code or Jupyter)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Open `01_data_preprocessing_and_eda.ipynb` in VS Code or Jupyter and run all cells from the project root. The notebook writes the cleaned data and EDA outputs into `outputs/`.

Then train models:

```bash
python src/train_models.py
```

Run the app:

```bash
streamlit run app.py
```

Optional MLflow UI:

```bash
mlflow ui
```

## Modeling assumptions and limitations

1. **Investment classification target:** The dataset has no pre-existing `Good_Investment` column. The notebook creates a proxy label: a property is labelled 1 when its recalculated price per square foot is at or below the median for its city. This identifies relative affordability, not actual profit, return on investment, or future appreciation.
2. **Leakage prevention:** The classifier excludes current price and price-per-square-foot columns because the label is derived from price per square foot.
3. **Five-year value:** The CSV has no historical or future transaction target. The regression model predicts current price from property features; future value is a scenario calculated as `predicted_current_price * (1 + annual_growth_rate) ** years`. The default 8% is the illustrative assumption given in the project brief, not a validated market forecast. The Streamlit app allows users to change this assumption.
4. **Price per square foot:** `Price_in_Lakhs` is converted to rupees (`lakhs × 100,000`) before dividing by square feet. The supplied `Price_per_SqFt` field appears inconsistent with this calculation, so its original value is retained as `Price_per_SqFt_Original`.
5. **Time trends:** The dataset has no transaction date. True price trends over time cannot be inferred; the notebook compares localities cross-sectionally instead.
6. **Results:** No performance values are hard-coded. Run the notebook and training script to produce metrics from the actual data. Accuracy or confidence should not be interpreted as investment certainty.

## Suggested 10-day delivery plan

1. Dataset understanding and audit
2. Cleaning and validation
3. EDA questions 1–10
4. EDA questions 11–20 and business insights
5. Feature engineering and target-definition review
6. Classification model comparison and evaluation
7. Regression model evaluation and forecasting assumptions
8. MLflow experiment tracking and explainability
9. Streamlit UI and testing
10. Documentation, GitHub, demo, and presentation

## Main evaluation metrics

- Classification: accuracy, F1-score, ROC-AUC, classification report
- Regression: MAE, RMSE, R²
- Experiment tracking: model parameters, metrics, and model artifacts in MLflow
