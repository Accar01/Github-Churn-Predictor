#contain the machine learning logic
"""
train the model
save the model
load the model
make predictions
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

# These are the selected features from Step 5.
# If your notebook printed a different SELECTED_FEATURES list, replace this list.
SELECTED_FEATURES = [
    "days_since_last_activity",
    "account_age_days",
    "language_count",
    "has_bio",
    "has_company",
    "has_location",
    "recent_event_count",
]


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = APP_DIR / "model.pkl"


def get_feature_file_path() -> Path:
    """
    Find the github_features.csv file.
    Works locally and inside Docker.
    """

    possible_paths = [
        Path("/data/processed/github_features.csv"),
        PROJECT_ROOT / "data" / "processed" / "github_features.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find github_features.csv. "
        "Run Step 4 first: docker-compose run --rm churn-api python features.py"
    )


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed feature dataset and separate X and y.
    X = selected features
    y = churned label
    """

    feature_path = get_feature_file_path()
    df = pd.read_csv(feature_path)

    print(f"Loaded feature file: {feature_path}")
    print(f"Dataset shape: {df.shape}")

    if "churned" not in df.columns:
        raise ValueError("The dataset must contain a 'churned' column.")

    missing_features = [
        feature for feature in SELECTED_FEATURES if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "These selected features are missing from github_features.csv: "
            f"{missing_features}"
        )

    X = df[SELECTED_FEATURES].copy()
    y = df["churned"].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The churned column has only one class. "
            "You need both churned = 0 and churned = 1 users."
        )

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Train the final Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X, y)

    return model


def evaluate_model(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate the model.

    If the dataset is too small for a safe train/test split,
    train and evaluate on the full dataset only.
    """

    class_counts = y.value_counts()

    can_split = (
        len(X) >= 10
        and y.nunique() == 2
        and class_counts.min() >= 2
    )

    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        evaluation_type = "train_test_split"

    else:
        y_pred = model.predict(X)
        evaluation_type = "training_data_only_small_dataset"

    metrics = {
        "evaluation_type": evaluation_type,
        "accuracy": accuracy_score(y if not can_split else y_test, y_pred),
        "precision": precision_score(
            y if not can_split else y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y if not can_split else y_test,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y if not can_split else y_test,
            y_pred,
            zero_division=0,
        ),
    }

    return metrics


def save_model(model: RandomForestClassifier, metrics: dict) -> None:
    """
    Save the trained model and selected features in one file.
    """

    model_bundle = {
        "model": model,
        "selected_features": SELECTED_FEATURES,
        "metrics": metrics,
    }

    joblib.dump(model_bundle, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


def load_saved_model() -> dict:
    """
    Load the saved model bundle.
    Used by main.py.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run: .\\.venv\\Scripts\\python.exe app\\model.py"
        )

    return joblib.load(MODEL_PATH)


def main() -> None:
    """
    Train and save the final model.
    """

    X, y = load_training_data()

    print("\nSelected features:")
    for feature in SELECTED_FEATURES:
        print(f"- {feature}")

    print("\nClass balance:")
    print(y.value_counts())

    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)

    # Train one final time on all available data before saving.
    final_model = train_model(X, y)

    save_model(final_model, metrics)

    print("\nModel evaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
    