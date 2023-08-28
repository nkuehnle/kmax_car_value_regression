# Wrangling/math/stats
import pandas as pd
import numpy as np


def update_imputed_vals(
        df: pd.DataFrame,
        col: str,
        imputed_vals: np.ndarray,
        imputed_vals_mask: np.ndarray
        ):
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    col : str
        _description_
    imputed_vals : np.ndarray
        _description_
    imputed_vals_mask : np.ndarray
        _description_
    """
    imp_indicator_col = f"{col}_imputed"
    if imp_indicator_col not in df.columns:
        df[f"{col}_imputed"] = imputed_vals_mask

    df.loc[imputed_vals_mask, col] = imputed_vals[imputed_vals_mask]