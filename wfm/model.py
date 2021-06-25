import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import h5py

from joblib import dump
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, make_scorer

from wfm.utils import get_display_and_numeric_data, _recall, recall
from wfm.constants import TARGET_COLUMN, X_COLUMNS, SPANISH_NAMES


logger = logging.getLogger(__name__)


def model_and_explanation(
    input_data,
    model_parameters,
    output_path,
    images_path,
    test_size=0.25,
    split_random_state=42,
    model_random_state=42,
):

    logger.info("Getting and spliting data.")
    # --- Design matrix and target vector ---
    model_objective = "binary"
    X_display, y_display, X, y = get_display_and_numeric_data(
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
    logger.info("Fitting the model.")
    # breakpoint()
    model = lgb.LGBMClassifier(**model_parameters, random_state=model_random_state)
    _ = model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=recall,

    )
    dump(model, output_path / "model.joblib")

    # --- Model Evaluation  ---
    logger.info("Classification Report.")
    y_pred = model.predict(X_test)
    logger.info("\n" +
        classification_report(
            y_test,
            y_pred,
        )
    )
    cross_validation_summary(model, X, y, images_path)

    # --- Model Explanation ---
    logger.info("Model Explanation using Shap values.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    with h5py.File(output_path / "shap_values.h5", 'w') as hf:
        hf.create_dataset("shap_values",  data=shap_values)
    # Summary Plot
    summary_plot(X, model, shap_values, images_path)
    # Dependence Plots
    dependence_plot(X, X_display, model, shap_values, images_path)

    return X_display, y_display, X, y, model, shap_values


def cross_validation_summary(model, X, y, images_path):
    recall_score = make_scorer(_recall, greater_is_better=True)
    cv_model = cross_validate(
        model,
        X,
        y,
        scoring=recall_score,
        cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=30, random_state=42),
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1
    )
    cv_scores = pd.DataFrame(
        {
            "Train": cv_model["train_score"],
            "Test": cv_model["test_score"]
        }
    )
    plt.figure(figsize=(14, 12))
    sns.boxplot(data=cv_scores, orient='v', color='cyan', saturation=0.5)
    plt.ylabel("Score")
    plt.title(f"Accuracy for Train and Test sets using 30 Repeated Stratified 4-Fold")
    plt.tight_layout()
    plt.savefig(images_path / f"cv_score.png", dpi=300)
    plt.show()
    plt.close()


def summary_plot(X, model, shap_values, images_path):
    logger.info("SHAP Summary Plot.")
    damage_idx = model.classes_.searchsorted(1)  # It should always be 1
    shap.summary_plot(
        shap_values[damage_idx],
        features=X,
        # feature_names=#TODO,
        max_display=X.shape[1],
        show=False
    )

    fig = plt.gcf()
    fig.suptitle(
        f"SHAP Summary Plot for LightGBM binary model.",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(images_path / f"shap_summary.png", dpi=300)
    plt.show()
    plt.close()


def dependence_plot(X, X_display, model, shap_values, images_path):
    damage_idx = model.classes_.searchsorted(1)
    for col in X.columns:
        logger.info(f"SHAP Depence Plot for {col}")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.dependence_plot(
            ind=col,
            shap_values=shap_values[damage_idx],
            features=X,
            display_features=X_display,
            interaction_index=None,
            # title=model.classes_[damage_idx],
            ax=ax,
            show=False
        )
        fig.suptitle(
            f"SHAP dependence plot for '{SPANISH_NAMES[col]}' feature",
            fontsize=12
        )
        fig.tight_layout()
        fig.savefig(images_path / f"shap_dependence_{col}.png", dpi=300)
        plt.close()
