import seaborn as sns

from pathlib import Path

import wfm

sns.set_context("paper")
sns.set_style("whitegrid")


def main(path):
    # --- Load ---
    path = Path(__file__).resolve().parent
    input_path = path / "data" / "input"
    # wfm.preprocessing.validate_columns(input_path)
    output_path = path / "output"
    images_path = path / "images"
    model_path = path / "model"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    input_data = wfm.preprocessing.get_input_data(input_path)

    # --- Exploratory Data Analysis ---
    # wfm.profile(input_data, output_path)
    # wfm.eda(input_data, images_path)
    wfm.model(input_data, model_path, images_path)


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent
    main(path)