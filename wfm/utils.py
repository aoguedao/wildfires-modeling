import pandas as pd

from wfm.constants import BINARY_TARGET_VALUES

def get_cat_code_dict(df, col):
    """Returns a dict to pase a categorical column to another categorical but using integer codes in order to use it with shap functions.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column to transform    

    Returns
    -------
    dict
        Original names as key and codes as values.
    """
    d = dict(
        zip(
            df[col],
            df[col].cat.codes.astype("category")
        )
    )
    return d


def get_display_and_numeric_data(
    input_data,
    X_COLUMNS,
    TARGET_COLUMN,
    model_objective="multiclass"
):
    # --- Data for display ---
    X_display = (
        input_data.loc[:, X_COLUMNS]
        .pipe(lambda x: pd.DataFrame(x))
    )
    y = input_data[TARGET_COLUMN]
    if model_objective == "binary":
        y = y.map(BINARY_TARGET_VALUES)
        # y = y.eq("Dañada").astype(int)

    # --- Data for algorithm ---
    X = X_display.copy()
    cat_codes = {
        col: X_display.pipe(get_cat_code_dict, col=col)
        for col in X_display.select_dtypes("category")
    }
    for col, d in cat_codes.items():
        X.loc[:, col] = X[col].map(d)

    return X_display, X, y