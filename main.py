import logging
import click
import seaborn as sns

from pathlib import Path

import wfm

sns.set_context("paper")
sns.set_style("whitegrid")


# logging.basicConfig(
#     # filename='HISTORYlistener.log',
#     # level=logging.DEBUG,
#     format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
# )
logger = logging.getLogger(__name__)


MODEL_PARAMETERS = {
    "multiclass":{
        "learning_rate": 0.05,
        'objective': "multiclass",
        'num_class':3,
        'metric': "multi_logloss",
        "verbose": -1,
    },
    "binary": {
        "learning_rate": 0.05,
        'objective': "binary",
        "metric": "binary_logloss",
        "verbose": -1,
    }
}
@click.command()
@click.option("--input_path", default=None, type=str)
@click.option("--model_parameters", default=MODEL_PARAMETERS, type=dict)
@click.option("--output_path", default=None, type=str)
@click.option("--images_path", default=None, type=str)
def wfm_main(
    input_path,
    model_parameters,
    output_path,
    images_path
):
    # --- Load ---
    path = Path(__file__).resolve().parent
    if input_path is None:
        input_path = path / "data" / "input"
    input_path.mkdir(parents=True, exist_ok=True)
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
    logging.info(f"Inputh path: {input_path}")
    logging.info(f"Output path: {output_path}")
    logging.info(f"Images path: {images_path}")
    logging.info(f"Design columns: {wfm.X_COLUMNS}")
    # Data
    input_data = wfm.preprocessing.get_input_data(input_path)
    input_data.to_excel(output_path / "input_data.xlsx", index=False)

    # --- Exploratory Data Analysis ---
    # wfm.profile(input_data, output_path)
    # wfm.eda(input_data, output_path, images_path)

    # --- Model and Explanation ---
    # Multiclass
    logger.info("Multiclass classification model")
    multiclass_output_path = output_path / "multiclass"
    multiclass_output_path.mkdir(parents=True, exist_ok=True)
    multiclass_images_path = images_path / "multiclass"
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
    binary_images_path = images_path / "binary"
    binary_images_path.mkdir(parents=True, exist_ok=True)
    wfm.model_and_explanation(
        input_data,
        model_parameters["binary"],
        binary_output_path,
        binary_images_path
    )


if __name__ == "__main__":
    wfm_main()