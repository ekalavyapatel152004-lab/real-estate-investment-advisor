"""
Train the real-estate investment screening classifier and current-price regressor.

Run from the project root:
    python src/train_models.py

Important:
- Good_Investment is a relative screening label: a property's price per sq ft
  is at or below its city's median. It is NOT verified investment profit.
- The regression model predicts CURRENT listing price, not actual future value.
- The app's five-year value is a scenario based on an assumed appreciation rate.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "cleaned_housing_data.csv"
RAW_DATA = ROOT / "data" / "india_housing_prices.csv"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset():
    """Load the cleaned file first, then fall back to the original CSV."""
    if DATA.exists():
        print(f"Loading cleaned dataset: {DATA}")
        data = pd.read_csv(DATA)
    elif RAW_DATA.exists():
        print(f"Cleaned file not found. Loading raw dataset: {RAW_DATA}")
        data = pd.read_csv(RAW_DATA)
    else:
        raise FileNotFoundError(
            "Dataset not found. Expected outputs/cleaned_housing_data.csv "
            "or data/india_housing_prices.csv."
        )

    data.columns = data.columns.astype(str).str.strip()
    return data


def prepare_dataset(data):
    """Validate values and create a consistent city-relative target label."""
    required = {"City", "Size_in_SqFt", "Price_in_Lakhs"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    for column in ["Size_in_SqFt", "Price_in_Lakhs"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Recalculate price per sq ft in rupees to avoid raw/cleaned unit mismatch.
    data["Price_per_SqFt"] = (
        data["Price_in_Lakhs"] * 100000 / data["Size_in_SqFt"].where(data["Size_in_SqFt"] > 0)
    )

    data["City"] = data["City"].astype("string").str.strip()
    data = data.replace([np.inf, -np.inf], np.nan)

    # Remove records that cannot produce a valid investment screening label.
    data = data.dropna(
        subset=["City", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt"]
    ).copy()
    data = data[
        (data["City"] != "")
        & (data["Size_in_SqFt"] > 0)
        & (data["Price_in_Lakhs"] > 0)
        & (data["Price_per_SqFt"] > 0)
    ].copy()

    if data.empty:
        raise ValueError("No valid rows remain after checking city, size, and price.")

    # Always regenerate the label so old invalid or inconsistent labels cannot pass through.
    city_medians = data.groupby("City")["Price_per_SqFt"].median()
    data["City_Median_Price_per_SqFt"] = data["City"].map(city_medians)
    data["Good_Investment"] = (
        data["Price_per_SqFt"] <= data["City_Median_Price_per_SqFt"]
    ).astype("int8")

    counts = data["Good_Investment"].value_counts().sort_index()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError(
            "Classification requires at least two records in each target class. "
            f"Current class counts: {counts.to_dict()}"
        )

    # Save a small audit summary for project documentation.
    audit = {
        "valid_rows_used": int(len(data)),
        "rows_removed_or_excluded": int(250000 - len(data)) if len(data) <= 250000 else None,
        "good_investment_rule": (
            "1 when Price_per_SqFt is at or below the median Price_per_SqFt "
            "for that City; otherwise 0."
        ),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "price_per_sqft_unit": "INR per square foot",
        "warning": "This is a relative price screening label, not observed profitability.",
    }
    (OUT / "target_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print(f"Valid rows used: {len(data):,}")
    print(f"Classification target counts: {counts.to_dict()}")
    return data


def make_preprocessor(features):
    """Impute and scale numeric fields; impute and one-hot encode categories."""
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=2,
                        max_categories=50,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No numeric or categorical model features were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_model_features(data):
    """Remove identifiers, targets, and price-derived columns from model inputs."""
    excluded = {
        "ID",
        "Good_Investment",
        "Good_Investment_Label_Definition",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Price_per_SqFt_Original",
        "City_Median_Price_per_SqFt",
    }
    features = data.drop(
        columns=[column for column in excluded if column in data.columns]
    ).copy()

    # Avoid passing completely empty columns into SimpleImputer.
    features = features.dropna(axis=1, how="all")
    if features.empty:
        raise ValueError("No usable feature columns remain after exclusions.")
    return features


def train_classifier(data):
    """Train and evaluate the relative investment-screening classifier."""
    X = get_model_features(data)
    y = data["Good_Investment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=14,
                    min_samples_leaf=5,
                    max_features=0.5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining investment screening classifier...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    joblib.dump(pipeline, MODEL_DIR / "investment_classifier.joblib")
    (OUT / "classification_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUT / "classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0),
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    ).to_csv(OUT / "confusion_matrix.csv")

    try:
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        values = pipeline.named_steps["model"].feature_importances_
        importance = pd.DataFrame({"Feature": names, "Importance": values})
        importance.sort_values(
            "Importance", ascending=False
        ).to_csv(OUT / "classification_feature_importance.csv", index=False)
    except Exception as exc:
        print(f"Feature importance export skipped: {exc}")

    return pipeline, metrics


def train_regressor(data):
    """Train a model for current listing price; it is not a future-price model."""
    X = get_model_features(data)
    y = data["Price_in_Lakhs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=40,
                    max_depth=14,
                    min_samples_leaf=5,
                    max_features=0.5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nTraining current-price regression model...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "mae_lakhs": float(mean_absolute_error(y_test, predictions)),
        "rmse_lakhs": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    # A median-only baseline provides a simple reference point.
    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    metrics.update(
        {
            "baseline_mae_lakhs": float(
                mean_absolute_error(y_test, baseline_predictions)
            ),
            "baseline_rmse_lakhs": float(
                np.sqrt(mean_squared_error(y_test, baseline_predictions))
            ),
            "baseline_r2": float(r2_score(y_test, baseline_predictions)),
        }
    )

    joblib.dump(pipeline, MODEL_DIR / "current_price_regressor.joblib")
    (OUT / "regression_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    return pipeline, metrics


def log_to_mlflow(classifier, classifier_metrics, regressor, regressor_metrics):
    """Track both runs and attempt to register both sklearn pipelines."""
    try:
        import mlflow
        import mlflow.sklearn

        # Store the tracking database inside this project folder.
        tracking_db = (ROOT / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
        mlflow.set_experiment("real-estate-investment-advisor")

        models_to_log = [
            (
                "investment-classifier-random-forest",
                classifier,
                classifier_metrics,
                "InvestmentClassifier",
            ),
            (
                "current-price-regressor-random-forest",
                regressor,
                regressor_metrics,
                "CurrentPriceRegressor",
            ),
        ]

        for run_name, model, metrics, registered_name in models_to_log:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags(
                    {
                        "project": "Real Estate Investment Advisor",
                        "target_type": (
                            "relative investment screening"
                            if registered_name == "InvestmentClassifier"
                            else "current listing price"
                        ),
                    }
                )
                mlflow.log_params(
                    {
                        "algorithm": "RandomForest",
                        "test_size": TEST_SIZE,
                        "random_state": RANDOM_STATE,
                        "numeric_scaling": "StandardScaler",
                        "categorical_encoding": "OneHotEncoder",
                    }
                )
                mlflow.log_metrics(metrics)

                # Keep the local joblib artifacts and register the sklearn model.
                model_file = (
                    MODEL_DIR / "investment_classifier.joblib"
                    if registered_name == "InvestmentClassifier"
                    else MODEL_DIR / "current_price_regressor.joblib"
                )
                mlflow.log_artifact(str(model_file))

                try:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name="model",
                        registered_model_name=registered_name,
                        # These are the specific sklearn/numpy types used by
                        # the RandomForest pipelines trained by this script.
                        # Trust them only because these model artifacts were
                        # created locally by this project.
                        skops_trusted_types=[
                            "numpy.dtype",
                            "sklearn.tree._tree.Tree",
                        ],
                    )
                    print(f"Registered MLflow model: {registered_name}")
                except Exception as registration_error:
                    print(
                        f"Model registration failed for {registered_name}: "
                        f"{registration_error}"
                    )
                    print(
                        "The experiment run and local joblib model are still saved."
                    )

        print(f"MLflow tracking database: {ROOT / 'mlflow.db'}")

    except ImportError:
        print(
            "\nMLflow is not installed in this Python environment. "
            "Models and evaluation files were saved locally, but MLflow "
            "tracking/registration was skipped. Install it with: "
            "python -m pip install mlflow"
        )
    except Exception as exc:
        print(f"\nMLflow tracking could not complete: {exc}")
        print("The trained local model files and evaluation outputs are saved.")


def main():
    data = prepare_dataset(load_dataset())

    classifier, classifier_metrics = train_classifier(data)
    regressor, regressor_metrics = train_regressor(data)

    summary = {
        "classification": classifier_metrics,
        "regression_current_price": regressor_metrics,
        "notes": {
            "classification_target": (
                "Price per sq ft at or below city median; relative screening only."
            ),
            "regression_target": "Current listing price in lakhs.",
            "five_year_value": (
                "Not learned from historical outcomes. If shown in the app, "
                "it is a scenario based on an assumed annual appreciation rate."
            ),
        },
    }
    (OUT / "model_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    log_to_mlflow(
        classifier,
        classifier_metrics,
        regressor,
        regressor_metrics,
    )

    print("\nClassification metrics:", classifier_metrics)
    print("Current-price regression metrics:", regressor_metrics)
    print("\nSaved models:")
    print(MODEL_DIR / "investment_classifier.joblib")
    print(MODEL_DIR / "current_price_regressor.joblib")
    print("\nImportant: the five-year value is an assumed-rate scenario,")
    print("not a machine-learning prediction of observed future prices.")


if __name__ == "__main__":
    main()
