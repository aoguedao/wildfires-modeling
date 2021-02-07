import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import h5py

from pathlib import Path
from datetime import date
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from wfm.utils import get_display_and_numeric_data
from wfm.constants import TARGET_COLUMN, X_COLUMNS, SPANISH_NAMES


logger = logging.getLogger(__name__)

MODEL_PARAMS = {
#     "max_bin": 512,
    "learning_rate": 0.05,
#     "boosting_type": "gbdt",
    'objective': 'multiclass',
    'num_class':3,
    'metric': 'multi_logloss',
    # "num_leaves": 10,
    "verbose": -1,
    # "min_data": 100,
}


def model(
    input_data,
    model_path,
    images_path,
    test_size=0.3,
    split_random_state=42,
    model_params=MODEL_PARAMS,
    model_random_state=42,
):

    # --- Design matrix and target vector ---
    X_display, X, y = get_display_and_numeric_data(
        input_data,
        X_COLUMNS,
        TARGET_COLUMN,
    )
    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=split_random_state
    )

    # --- LightGBM Model  ---
    model = lgb.LGBMClassifier(**model_params, random_state=model_random_state)
    _ = model.fit(X_train, y_train)

    # --- Model Evaluation  ---
    y_pred = model.predict(X_test)
    print(
        classification_report(
            y_test,
            y_pred,
        )
    )

    # Some utiles
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    model_path = model_path / today_str
    model_path.mkdir(parents=True, exist_ok=True)
    dump(model, model_path / "model.joblib")
    
    # --- Model Explanation ---
    # Shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    with h5py.File(model_path / "shap_values.h5", 'w') as hf:
        hf.create_dataset("shap_values",  data=shap_values)

    # Summary Plot
    shap_images_path = images_path / "shap" / today_str
    shap_images_path.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        shap_values,
        X,
        max_display=X.shape[1],
        class_names=model.classes_,
        show=False
    )
    plt.tight_layout()
    plt.savefig(shap_images_path / f"shap_summary.png", dpi=300)

    # Dependence Plots
    for col in X.columns:
        fig, axes = plt.subplots(ncols=3, figsize=(20, 5), sharey=True)
        for i, ax in enumerate(axes):
            shap.dependence_plot(
                col,
                shap_values[i],
                X,
                display_features=X_display,
                interaction_index=None,
                title=model.classes_[i],
                ax=ax,
                show=False
            )
        fig.suptitle(
            f"SHAP dependence plot for {SPANISH_NAMES[col]} feature",
            fontsize=16
        )
        fig.tight_layout()
        fig.savefig(shap_images_path / f"shap_dependence_{col}.png", dpi=300)
        plt.close()
