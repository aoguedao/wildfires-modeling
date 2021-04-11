import logging
import click
import seaborn as sns

from pathlib import Path

import wfm

sns.set_context("paper")
sns.set_style("whitegrid")


MODEL_PARAMETERS = {
    "multiclass":{
        'objective': "multiclassova",
        'num_class':3,
        'metric': "multi_logloss",
        "is_unbalance": True,
        # "learning_rate": 0.01,
        # "max_depth":12,
        "verbose": -1,
    },
    "binary": {
        'objective': "binary",
        "metric": "binary_logloss",
        "is_unbalance": True,
        "learning_rate": 0.01,
        # "max_depth":12,
        # "mini_data_per_leaf": 1,
        # "feature_fraction": 0.7,
        # "lambda_l1": 0.1,
        # "lambda_l2": 0.1,
        # "scale_pos_weight": 10,
        "verbose": -1,
    }
}
@click.command()
@click.option("--input_path", default=None, type=str)
@click.option("--model_parameters", default=MODEL_PARAMETERS, type=dict)
@click.option("--output_path", default=None, type=str)
@click.option("--images_path", default=None, type=str)
@click.option("--log", default=None, type=str)
@click.option('--eda', is_flag=True)
def wfm_main(
    input_path,
    model_parameters,
    output_path,
    images_path,
    log,
    eda
):

    if log is None:
        from datetime import datetime
        now = datetime.now()
        log = f"wildfire-modeling-{now}"
    logger = logging.getLogger("wfm")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"{log}.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    # --- Load ---
    path = Path(__file__).resolve().parent
    if input_path is None:
        input_path = path / "input"
    # wfm.preprocessing.validate_columns(input_path)
    if output_path is None:
        output_path = path / "output"
    else:
        output_path = Path(output_path)
    
    if images_path is None:
        images_path = path / "images"
    else:
        images_path = Path(images_path)
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Inputh path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Images path: {images_path}")
    logger.info(f"Design columns: {wfm.X_COLUMNS}")
    
    # Data
    input_data = wfm.preprocessing.get_input_data(input_path)
    input_data.to_excel(output_path / "input_data.xlsx", index=False)

    # --- Exploratory Data Analysis ---
    if eda:
        wfm.profile(input_data, output_path)
        wfm.eda(input_data, output_path, images_path)

    # --- Model and Explanation ---
    # Multiclass
    logger.info("Multiclass classification model")
    multiclass_output_path = output_path / "multiclass"
    multiclass_output_path.mkdir(parents=True, exist_ok=True)
    multiclass_images_path = multiclass_output_path / "images"
    multiclass_images_path.mkdir(parents=True, exist_ok=True)
    wfm.model_and_explanation(
        input_data,
        model_parameters["multiclass"],
        multiclass_output_path,
        multiclass_images_path
    )

    # Binary
    logger.info("Binary classification model")
    binary_output_path = output_path / "binary"
    binary_output_path.mkdir(parents=True, exist_ok=True)
    binary_images_path = binary_output_path / "images" 
    binary_images_path.mkdir(parents=True, exist_ok=True)
    wfm.model_and_explanation(
        input_data,
        model_parameters["binary"],
        binary_output_path,
        binary_images_path
    )


if __name__ == "__main__":
    wfm_main()