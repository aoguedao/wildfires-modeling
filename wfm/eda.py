import logging
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas_profiling import ProfileReport
from datetime import date
from itertools import permutations
from pandas.api.types import CategoricalDtype

from wfm.constants import TARGET_COLUMN, NUM_COLUMNS, CAT_COLUMNS, SPANISH_NAMES


logger = logging.getLogger(__name__)


def profile(input_data, profile_path):
    logger.info("Data profiling.")
    today = date.today()
    df = input_data.drop(columns="geometry")
    profile = ProfileReport(
        df,
        title="Estadística Descriptiva Incendios Edificios",
        explorative=True
    )
    profile.to_file(profile_path / f"edificaciones-profile-{today.strftime('%Y-%m-%d')}.html")
    for name, group in df.groupby("wildfire"):
        profile = ProfileReport(
            group,
            title=f"Estadística Descriptiva Incendio {name.title()}",
            explorative=True
        )
        profile.to_file(profile_path / f"edificaciones-{name}-profile-{today.strftime('%Y-%m-%d')}.html")


def eda(input_data, images_path):
    cat_histograms(input_data, images_path)
    num_histograms(input_data, images_path)
    num_scatters(input_data, images_path)
    correlation(input_data, images_path)
    # pairplot(input_data, images_path)


def cat_histograms(input_data, images_path):
    redable_target = SPANISH_NAMES[TARGET_COLUMN]
    for col in CAT_COLUMNS:
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
            hue=TARGET_COLUMN,
            hue_order=["Parcial", "Total", "Ninguno"],
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
        plt.savefig(images_path / f"cat_histogram_{col}.png")
        plt.close()


def num_histograms(input_data, images_path):
    df = (
        input_data.copy()#.fillna("N/A")
        # .astype({col: "category" for col in CAT_COLUMNS})
    )
    for col in NUM_COLUMNS:
        logger.info(f"Histogram {col}")
        redable_col = SPANISH_NAMES[col]
        redable_target = SPANISH_NAMES[TARGET_COLUMN]
        plt.figure(figsize=(10, 8))
        g = sns.histplot(
            df.sort_values(col),
            x=col,
            hue=TARGET_COLUMN,
            hue_order=["Parcial", "Total", "Ninguno"],
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
        plt.savefig(images_path / f"histogram_{col}.png")
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

    plt.figure(figsize=(20, 20))
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
    plt.xticks(rotation=50.4, horizontalalignment="left")
    plt.yticks(rotation=50.4, horizontalalignment="right")
    plt.title("Correlación entre variables numéricas")
    plt.tight_layout()
    plt.savefig(images_path / f"correlation.png")
    plt.close()