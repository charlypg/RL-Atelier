import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

def plot_from_dataframe(
    fig, ax,
    df: pd.DataFrame,
    x_key: str, y_key: str,
    fill_between: Optional[Tuple[str, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
):
    x = pd.to_numeric(df[x_key], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_key], errors="coerce").to_numpy()
    
    if fill_between is not None:
        ymin = pd.to_numeric(df[fill_between[0]], errors="coerce").to_numpy()
        ymax = pd.to_numeric(df[fill_between[1]], errors="coerce").to_numpy()
        ax.fill_between(x, ymin, ymax, alpha=0.2)
    
    ax.plot(x, y)
    ax.grid()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return fig, ax

def lineplot_from_dataframe(
    fig, ax,
    df: pd.DataFrame,
    x_key: str, y_key: str,
    hue: Optional[str] = None,
    errorbar: Optional[Tuple[str, float|int]] = ("pi", 50),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None
):
    sns.lineplot(
        data=df,
        x=x_key,
        y=y_key,
        hue=hue,
        ax=ax,
        errorbar=errorbar
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    return fig, ax

def group_data_from_path(
    path: str,
    csv_filename: str,
    method: Optional[str] = None
) -> pd.DataFrame:
    list_of_dataframes = []
    for subpath in os.listdir(path):
        if os.path.isdir(os.path.join(os.path.abspath(path), subpath)):
            list_of_dataframes.append(
                pd.read_csv(
                    os.path.join(path, subpath, csv_filename)
                )
            )
    df = pd.concat(list_of_dataframes, ignore_index=True)
    if method is not None:
        df["method"] = method
    return df

def group_data(path: str, csv_filename: str) -> pd.DataFrame:
    list_of_dataframes = []
    subs = os.listdir(path)
    for subpath in subs:
        abs_dir_subpath = os.path.join(os.path.abspath(path), subpath)
        if os.path.isdir(abs_dir_subpath):
            if "method.txt" in os.listdir(os.path.join(abs_dir_subpath)):
                with open(os.path.join(path, subpath, "method.txt"), "r") as f:
                    method = f.read()
            else:
                method = None
            list_of_dataframes.append(
                group_data_from_path(os.path.join(path, subpath), csv_filename, method)
            )
    df = pd.concat(list_of_dataframes, ignore_index=True)
    return df

if __name__ == "__main__":
    name_folder = "results"
    csv_filename = "eval.csv"

    path = os.path.join(".", name_folder)
    df = group_data(path, csv_filename)
    if "method" in df.keys():
        method = "method"
    else:
        method = None

    fig, ax = plt.subplots(figsize=(9, 5))
    fig, ax = lineplot_from_dataframe(
        fig, ax, df,
        x_key="step", y_key="return_median",
        hue=method,
        xlabel="Step",
        ylabel="Sum of rewards"
    )
    fig.savefig("sum_of_rewards.pdf")
