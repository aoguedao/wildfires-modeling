# Wildfires Modeling (WFM)

This project aims to determine most influential features regarding the damage suffered by houses caused by forest fires using multivariate statistical models.


## Methodology

* Exploratory Data Analysis
* Data Cleaning
* Feature Selection
* Binary Classification (['LightGBM'](https://lightgbm.readthedocs.io/))
* Feature Importance (['shap'](https://shap.readthedocs.io/))


## Setup

Clone this repository, move to the folder and run on your favourite environment ('conda', 'mamba', 'venv', 'docker', etc.) the following:
```
python -m pip install -e wfm
```
The flag `-e` mean this is a installation in developing mode, in order you can modify some parameters.


## Quick Start

You must have an input folder where each scenario must be another folder with georeferenced files inside, e.f. `.shp`.

For exploratory data analysis you can use the file `eda_cli.py` as follow
```
python eda_cli.py --input_path {YOUR_INPUT_PATH} --output_path {YOUR_OUTPUT_PATH}
```
Both arguments are optionals, for default input and output paths are `input` and `exploratory_data_analysis` respectively.

Same for data modeling using `main.py` as follow
```
python main.py --input_path {YOUR_INPUT_PATH} --output_path {YOUR_OUTPUT_PATH}
```
In this case, default values are 'input' and 'output' respectively.