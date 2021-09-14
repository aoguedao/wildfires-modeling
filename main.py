import logging
import click
import seaborn as sns

from pathlib import Path
from datetime import datetime

import wfm

sns.set_context("paper")
sns.set_style("whitegrid")


MODEL_PARAMETERS = {
    'objective': "binary",
    # "metric": "binary_logloss",
    # "is_unbalance": True,
    "learning_rate": 0.01,
    # "max_depth":12,
    # "mini_data_per_leaf": 1,
    # "feature_fraction": 0.7,
    # "lambda_l1": 0.1,
    # "lambda_l2": 0.1,
    "scale_pos_weight": 99,
    "verbose": -1,
}

@click.command()
@click.option("--input_path", default=None, type=str)
@click.option("--model_parameters", default=MODEL_PARAMETERS, type=dict)
@click.option("--output_path", default=None, type=str)
def wfm_main(
    input_path,
    model_parameters,
    output_path,
):

    # --- Load ---
    path = Path(__file__).resolve().parent
    if input_path is None:
        input_path = path / "input"
    if output_path is None:
        output_path = path / "output"
    else:
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logname = f"wildfire-modeling-{datetime.now()}"
    logger = logging.getLogger("wfm")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(str(output_path / f"{logname}.log"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Inputh path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Images path: {images_path}")
    logger.info(f"Design columns: {wfm.X_COLUMNS}")

    # Data
    input_data = wfm.preprocessing.get_input_data(input_path)
    input_data.to_excel(output_path / "input_data.xlsx", index=False)

    # --- Model and Explanation ---
    wfm.model_and_explanation(
        input_data,
        model_parameters,
        output_path,
        images_path
    )

    wfm.group_model_and_explanation(
        input_data,
        model_parameters,
        output_path,
        images_path,
    )

if __name__ == "__main__":
    wfm_main()