import os
from typing import NoReturn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CAT_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUM_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
REP_DIR = "reports"


def describe_num_cols(df: pd.DataFrame) -> NoReturn:
    desc = df[NUM_COLUMNS].describe()
    desc.to_csv(os.path.join(REP_DIR, "num.csv"))


def describe_cat_cols(df: pd.DataFrame) -> NoReturn:
    desc = df[CAT_COLUMNS].astype("category").describe()
    desc.to_csv(os.path.join(REP_DIR, "cat.csv"))


def pie_plot(df: pd.DataFrame) -> NoReturn:
    plt.pie(
        df.groupby(["sex", "target"]).size(),
        labels=["no disease", "disease", "no disease", "disease"],
        colors=["tab:red", "tab:red", "tab:blue", "tab:blue"],
        explode=(0, 0.05, 0, 0.05),
        autopct="%1.1f%%", radius=1.2, textprops={"fontsize": 14}
    )
    plt.savefig(os.path.join(REP_DIR, "pie_plot.png"))


def num_target_pairplot(df: pd.DataFrame) -> NoReturn:
    sns.pairplot(df, vars=NUM_COLUMNS, hue="target")
    plt.savefig(os.path.join(REP_DIR, "tgt_pairplot.png"))


def main():
    os.makedirs(REP_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join("data", "raw", "heart.csv"))
    describe_num_cols(df)
    describe_cat_cols(df)
    pie_plot(df)
    num_target_pairplot(df)


if __name__ == "__main__":
    plt.style.use("seaborn")
    main()
