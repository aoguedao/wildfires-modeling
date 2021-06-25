import logging
import click
import seaborn as sns

from pathlib import Path
from datetime import datetime

import wfm

sns.set_context("paper")
sns.set_style("whitegrid")


@click.command()
@click.option("--input_path", default=None, type=str)
@click.option("--output_path", default=None, type=str)
def main(
    input_path,
    output_path,
):

    # --- Load ---
    path = Path(__file__).resolve().parent
    if input_path is None:
        input_path = path / "input"
    # wfm.preprocessing.validate_columns(input_path)
    if output_path is None:
        output_path = path / "exploratory_data_analysis"
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
    logger.info(f"Design columns: {wfm.X_COLUMNS}")

    # Data
    input_data = wfm.preprocessing.get_input_data(input_path)
    wfm.profile(input_data, output_path)
    wfm.eda(input_data, output_path, images_path)


if __name__ == "__main__":
    main()