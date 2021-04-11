import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import h5py

from pathlib import Path
# from datetime import date
from joblib import dump
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import classification_report

from wfm.utils import get_display_and_numeric_data
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
    model_objective = model_parameters.get("objective")
    X_display, X, y = get_display_and_numeric_data(
        input_data,
        X_COLUMNS,
        TARGET_COLUMN,
        model_objective,
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
    model = lgb.LGBMClassifier(**model_parameters, random_state=model_random_state)
    _ = model.fit(X_train, y_train)
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
    # Cross Validation
    cv_model = cross_validate(
        model,
        X,
        y,
        cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=30, random_state=42),
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1
    )
    cv_scores = pd.DataFrame(
        {"Train": cv_model["train_score"],
        "Test": cv_model["test_score"]
        }
    )
    plt.figure(figsize=(10, 9))
    sns.boxplot(data=cv_scores, orient='v', color='cyan', saturation=0.5)
    plt.ylabel("Score")
    plt.title(f"Accuracy for Train and Test sets using Repeated Stratified KFold")
    plt.tight_layout()
    plt.savefig(images_path / f"cv_score.png", dpi=300)
    plt.show()
    plt.close()

    # --- Model Explanation ---
    logger.info("Model Explanation.")
    # Shap
    logger.info("Computing SHAP values.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    with h5py.File(output_path / "shap_values.h5", 'w') as hf:
        hf.create_dataset("shap_values",  data=shap_values)

    # Summary Plot
    summary_plot(X, model, shap_values, images_path)
    # Dependence Plots
    dependence_plot(X, X_display, model, shap_values, images_path)
    
    return X_display, X, y, model, shap_values


def summary_plot(X, model, shap_values, images_path):
    logger.info("SHAP Summary Plot.")
    nclasses = len(model.classes_)
    if nclasses > 2:
        shap.summary_plot(
            shap_values,
            X,
            max_display=X.shape[1],
            class_names=model.classes_,
            show=False
        )
    elif nclasses == 2:
        damage_idx = model.classes_.searchsorted("Dañada")
        shap.summary_plot(
            shap_values[damage_idx],
            X,
            max_display=X.shape[1],
            class_names=model.classes_[damage_idx],
            show=False
        )
    else:
        raise NotImplementedError
    fig = plt.gcf()
    fig.suptitle(
        f"SHAP Summary Plot for {model.get_params().get('objective')} model.",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(images_path / f"shap_summary.png", dpi=300)
    plt.show()
    plt.close()

    for i in range(len(shap_values)):
        model_class = model.classes_[i]
        shap.summary_plot(
            shap_values[i],
            X,
            max_display=X.shape[1],
            class_names=model_class,
            show=False
        )
        fig = plt.gcf()
        fig.suptitle(
            f"SHAP Summary Plot for {model.get_params().get('objective')} model and {model_class} class",
            fontsize=14
        )
        plt.tight_layout()
        plt.savefig(images_path / f"shap_summary_scatter_{model_class}.png", dpi=300)
        plt.show()
        plt.close()


def dependence_plot(X, X_display, model, shap_values, images_path):
    nclasses = len(model.classes_)
    for col in X.columns:
        logger.info(f"SHAP Depence Plot for {col}")
        if nclasses == 2:
            damage_idx = model.classes_.searchsorted("Dañada")
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.dependence_plot(
                col,
                shap_values[damage_idx],
                X,
                display_features=X_display,
                interaction_index=None,
                title=model.classes_[damage_idx],
                ax=ax,
                show=False
            )
        else:
            fig, axes = plt.subplots(ncols=nclasses, figsize=(7 * nclasses, 5), sharey=True)
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
            f"SHAP dependence plot for '{SPANISH_NAMES[col]}' feature",
            fontsize=14
        )
        fig.tight_layout()
        fig.savefig(images_path / f"shap_dependence_{col}.png", dpi=300)
        plt.close()