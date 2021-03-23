# import os
# import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from pprint import pprint

from wfm.constants import BUILDING_COLUMNS, FIX_BUILDING_COLUMN_NAMES, INPUT_COLUMNS, CAT_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def get_input_filepaths(input_path):
    p = Path(input_path).glob('**/Edificaciones.shp')
    files = [x for x in p if x.is_file()]
    return files


def read_building(filepath):
    logger.info(f"Reading {filepath} file.")
    df = (
        gpd.read_file(filepath)
        .rename(columns=lambda x: x.strip().lower())
        .rename(columns=FIX_BUILDING_COLUMN_NAMES)
        .pipe(add_missing_columns, columns=BUILDING_COLUMNS)
        # .loc[:, BUILDING_COLUMNS]
    )
    if set(BUILDING_COLUMNS) != set(df.columns):
        leftover_cols = df.columns.difference(BUILDING_COLUMNS)
        logger.warning(f"There are leftover columns: {leftover_cols}")
    return df


def add_missing_columns(df, columns):
    df = df.copy()
    current_cols = set(df.columns)
    missing_cols = set(columns) - current_cols
    for col in missing_cols:
        logger.warning(f"Adding column {col} with null values.")
        df[col] = np.nan
    return df


def get_buildings(input_path):
    filepaths = get_input_filepaths(input_path)
    df_dict = {f.parent.name.title(): read_building(f) for f in filepaths}
    return df_dict


def validate_columns(input_path):
    df_dict = get_buildings(input_path)
    df_val = (
        pd.concat(
            [pd.Series(1, index=df.columns, name=key) for key, df in df_dict.items()],
            axis=1
        )
        .fillna("")
        .sort_index()
    )
    pprint(df_val)


def get_input_data(input_path):
    df_dict = get_buildings(input_path)
    input_data = (
        pd.concat(df_dict)
        .droplevel(1)
        .rename_axis("wildfire")
        .reset_index()
        .loc[:, INPUT_COLUMNS]
        .astype({col: "category" for col in CAT_COLUMNS + [TARGET_COLUMN]})
    )
    return input_data
