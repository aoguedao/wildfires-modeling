import logging
import seaborn as sns

from datetime import date
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
        # 'num_class': 1,
        "metric": "binary_logloss",
        "verbose": -1,
    }
}


def wfm_main(
    input_path,
    model_parameters=MODEL_PARAMETERS,
    output_path=None,
    images_path=None
):
    # --- Load ---
    # wfm.preprocessing.validate_columns(input_path)
    if output_path is None:
        output_path = path / "output"
    if images_path is None:
        images_path = path / "images" 
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    # Data
    input_data = wfm.preprocessing.get_input_data(input_path)

    # --- Exploratory Data Analysis ---
    wfm.profile(input_data, output_path)
    wfm.eda(input_data, output_path, images_path)

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
    path = Path(__file__).resolve().parent
    input_path = path / "data" / "input"
    input_path.mkdir(parents=True, exist_ok=True)
    wfm_main(
        input_path=input_path,
        model_parameters=MODEL_PARAMETERS,
    )