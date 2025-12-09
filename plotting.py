import pandas as pd
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
