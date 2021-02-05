import logging
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from joblib import dump
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report

log = logging.getLogger(__name__)


def train(input_data, model_path):

    # --- Load data
    input_data = input_data.fillna("0")

    # --- Split data 
    split_random_state = 42
    X = (
        input_data.drop(columns=["nombre", "n_daño", "geometry"])
        .pipe(lambda df: pd.DataFrame(df))
    )
    y = input_data["n_daño"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=split_random_state
    )
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"{y_train.size} train records and {y_test.size} test records.")

    # --- Preprocessing
    categorical_columns = X.select_dtypes("object").columns.tolist()
    numerical_columns = X.select_dtypes("number").columns.tolist()
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        remainder="passthrough"
    )

    # --- Model
    model = make_pipeline(
        # imputer,
        preprocessor,
        RandomForestClassifier(
            n_estimators=10,
            n_jobs=-1,
            random_state=42,
        )
    )

    _ = model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    print(
        classification_report(
            y_test,
            y_pred_test,
        )
    )
    now = datetime.now()
    dump(
        model,
        model_path / f"model-{now.strftime('%Y-%m-%d-%H:%M')}.joblib"
    ) 

# if __name__ == "__main__":
#     path = Path(__file__).resolve().parent.parent
#     input_path = path / "data" / "input"
#     model_path = path / "model"
#     input_path.mkdir(parents=True, exist_ok=True)
#     model_path.mkdir(parents=True, exist_ok=True)
#     train(input_path, model_path)