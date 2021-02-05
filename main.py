import seaborn as sns

from pathlib import Path

import wfm

sns.set_context("paper")
sns.set_style("whitegrid")

# --- Load ---
path = Path(__file__).resolve().parent
input_path = path / "data" / "input"
output_path = path / "output"
images_path = path / "images"
output_path.mkdir(parents=True, exist_ok=True)
images_path.mkdir(parents=True, exist_ok=True)
input_data = wfm.preprocessing.get_input_data(input_path)

# --- Exploratory Data Analysis ---
# wfm.validate_columns(input_path)
# wfm.profile(input_data, output_path)
wfm.eda(input_data, images_path)