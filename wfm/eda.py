import logging
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas_profiling import ProfileReport
from itertools import permutations
from pandas.api.types import CategoricalDtype
# from seaborn.palettes import husl_palette

from wfm.constants import TARGET_COLUMN, NUM_COLUMNS, CAT_COLUMNS, SPANISH_NAMES


logger = logging.getLogger(__name__)


def profile(input_data, output_path):
    logger.info("Data profiling.")
    profile_path = output_path / "profiles"
    profile_path.mkdir(parents=True, exist_ok=True)
    df = input_data.drop(columns="geometry")
    profile = ProfileReport(
        df,
        title="Estadística Descriptiva Incendios Edificios",
        explorative=True
    )
    profile.to_file(profile_path / "input-data-profile.html")
    for name, group in df.groupby("wildfire"):
        profile = ProfileReport(
            group,
            title=f"Estadística Descriptiva Incendio {name.title()}",
            explorative=True
        )
        profile.to_file(profile_path / f"data-{name}-profile.html")


def eda(input_data, output_path, images_path):
    descriptive(input_data, output_path)
    cat_histograms(input_data, images_path)
    num_histograms(input_data, images_path)
    num_scatters(input_data, images_path)
    cat_histograms(input_data, images_path, histogram_hue="wildfire")
    num_histograms(input_data, images_path, histogram_hue="wildfire")
    correlation(input_data, images_path)
    kde(input_data, images_path)
    kde(input_data, images_path, kde_hue="wildfire")
    # pairplot(input_data, images_path)  # Deprecated


def descriptive(input_data, output_path):
    input_data = input_data.drop(columns="geometry")
    descriptive_all = input_data.describe(include="all").T
    descriptive_grouped = (
        pd.concat(
            {
                name: group.describe(include="all").drop(columns="wildfire").T
                for name, group in input_data.groupby("wildfire")
            }
        )
    )
    descriptive_target = (
        pd.concat(
            {
                name: group.describe(include="all").drop(columns="n_daño").T
                for name, group in input_data.groupby("n_daño")
            }
        )
    )
    target_balance = (
        input_data["n_daño"]
        .pipe(
            lambda x:
            pd.concat(
                {
                    "n_records": x.value_counts(dropna=False),
                    "perc_records": x.value_counts(normalize=True, dropna=False) * 100
                },
                axis=1
            )
        )
    )
    target_per_wildfire = input_data.pivot_table(
        index="wildfire",
        columns="n_daño",
        values="id",
        aggfunc="count",
        margins=True
    )
    with pd.ExcelWriter(output_path / "descriptive_statistics.xlsx") as writer:
        descriptive_all.to_excel(writer, sheet_name="All")
        descriptive_grouped.to_excel(writer, sheet_name="Per Wildfire")
        descriptive_target.to_excel(writer, sheet_name="Per Target Label")
        target_per_wildfire.to_excel(writer, sheet_name="Per Target and Wildfire")
        target_balance.to_excel(writer, sheet_name="Target Balance")


def cat_histograms(input_data, images_path, histogram_hue=None):
    if histogram_hue is None:
        histogram_hue = TARGET_COLUMN
    histogram_path = images_path / f"histogram_{histogram_hue}"
    histogram_path.mkdir(parents=True, exist_ok=True)
    input_data = (
        input_data.astype({col: CategoricalDtype(ordered=True) for col in CAT_COLUMNS + [TARGET_COLUMN]})
    )
    redable_target = SPANISH_NAMES[histogram_hue]
    for col in CAT_COLUMNS:
        if histogram_hue == col:
            continue
        if histogram_hue == TARGET_COLUMN:
            histogram_hue_order = ["Parcial", "Total", "Ninguno"]
        else:
            histogram_hue_order = None
        logger.info(f"Categorical histogram {col}")
        redable_col = SPANISH_NAMES[col]
        df = (
            input_data.loc[lambda x: x[col].sort_values().index, :]
            .astype(object)
            .fillna({col: "N/A"})
        )
        plt.figure(figsize=(10, 8))
        g = sns.histplot(
            df,
            x=col,
            hue=histogram_hue,
            hue_order=histogram_hue_order,
            multiple="dodge",
            shrink=.8,
            palette="Set2",
            legend=True,
        )
        g.get_legend().set_title(redable_target)
        plt.xlabel(redable_col)
        plt.ylabel("Conteo")
        plt.xticks(rotation=50.4, horizontalalignment="left")
        plt.title(f"Histograma de {redable_col} por {redable_target}")
        plt.tight_layout()
        plt.savefig(histogram_path / f"cat_histogram_{col}.png")
        plt.close()


def num_histograms(input_data, images_path, histogram_hue=None):
    if histogram_hue is None:
        histogram_hue = TARGET_COLUMN
    if histogram_hue == TARGET_COLUMN:
        histogram_hue_order = ["Parcial", "Total", "Ninguno"]
    else:
        histogram_hue_order = None
    histogram_path = images_path / f"histogram_{histogram_hue}"
    histogram_path.mkdir(parents=True, exist_ok=True)
    df = input_data.copy()
    redable_target = SPANISH_NAMES[histogram_hue]
    for col in NUM_COLUMNS:
        if histogram_hue == col:
            continue
        logger.info(f"Histogram {col}")
        redable_col = SPANISH_NAMES[col]
        plt.figure(figsize=(10, 8))
        g = sns.histplot(
            df.sort_values(col),
            x=col,
            hue=histogram_hue,
            hue_order=histogram_hue_order,
            multiple="stack",
            # shrink=.8,
            palette="Set2",
            legend=True,
        )
        g.get_legend().set_title(redable_target)
        plt.xlabel(redable_col)
        plt.ylabel("Conteo")
        plt.xticks(rotation=50.4, horizontalalignment="left")
        plt.title(f"Histograma de {redable_col} por {redable_target}")
        plt.tight_layout()
        plt.savefig(histogram_path / f"histogram_{col}.png")
        plt.close()


def pairplot(input_data, images_path):
    logger.info("Plotting numeric columns pairplot.")
    sns.pairplot(
        input_data[NUM_COLUMNS + [TARGET_COLUMN]],
        hue=TARGET_COLUMN,
        diag_kind="hist",
        corner=True,
        palette="Set2",
        height=40,
    )
    plt.title(f"Pairplot variables numéricas")
    plt.tight_layout()
    plt.savefig(images_path / f"numeric_pairplot.png")
    plt.close()


def num_scatters(input_data, images_path):
    scatter_path = images_path / "scatter"
    scatter_path.mkdir(parents=True, exist_ok=True)
    input_data = (
        input_data.astype({col: CategoricalDtype(ordered=True) for col in CAT_COLUMNS + [TARGET_COLUMN]})
    )
    for colx, coly in permutations(NUM_COLUMNS, 2):
        logger.info(f"ScatterPlot {coly} vs {colx}")
        redable_x = SPANISH_NAMES[colx]
        redable_y = SPANISH_NAMES[coly]
        redable_target = SPANISH_NAMES[TARGET_COLUMN]
        plt.figure(figsize=(8, 8))
        g = sns.scatterplot(
            data=input_data,
            x=colx,
            y=coly,
            hue=TARGET_COLUMN,
            hue_order=["Parcial", "Total", "Ninguno"],
            linewidth=0,
            alpha=.7
        )
        g.get_legend().set_title(redable_target)
        plt.xlabel(redable_x)
        plt.ylabel(redable_y)
        # plt.xticks(rotation=50.4, horizontalalignment="left")
        plt.title(f"Gráfico de {redable_y} v/s {redable_x} por {redable_target}")
        plt.tight_layout()
        plt.savefig(scatter_path / f"scatterplot_{coly}_vs_{colx}.png")
        plt.close()


def correlation(input_data, images_path):
    logger.info("Correlation")
    corr = input_data[NUM_COLUMNS].corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .3}
    )
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.yticks(rotation=45, horizontalalignment="right")
    plt.title("Correlación entre variables numéricas")
    plt.tight_layout()
    plt.savefig(images_path / f"correlation.png", dpi=300)
    plt.close()


def kde(input_data, images_path, kde_hue=None):
    if kde_hue is None:
        kde_hue = TARGET_COLUMN
    if kde_hue == TARGET_COLUMN:
        kde_hue_order = ["Parcial", "Total", "Ninguno"]
    else:
        kde_hue_order = None
    kde_path = images_path / f"kde_{kde_hue}"
    kde_path.mkdir(parents=True, exist_ok=True)
    df = input_data.copy()
    redable_target = SPANISH_NAMES[kde_hue]
    for col in NUM_COLUMNS:
        if kde_hue == col:
            continue
        logger.info(f"KDE {col}")
        redable_col = SPANISH_NAMES[col]
        plt.figure(figsize=(10, 8))
        g = sns.kdeplot(
            data=df.sort_values(col),
            x=col,
            hue=kde_hue,
            hue_order=kde_hue_order,
            common_norm=False,
            common_grid=True,
            palette="Set2",
            fill=True,
            alpha=0.5,
            legend=True,
        )
        g.get_legend().set_title(redable_target)
        plt.xlabel(redable_col)
        plt.ylabel("Conteo")
        plt.xticks(rotation=50.4, horizontalalignment="left")
        plt.title(f"Kernel Density Estimate de {redable_col} por {redable_target}")
        plt.tight_layout()
        plt.savefig(kde_path / f"kde_{col}.png")
        plt.close()
