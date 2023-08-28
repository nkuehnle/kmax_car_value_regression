import pandas as pd
import numpy as np


def merge_rare_makes(df: pd.DataFrame, n: int = 15):
    make_counts = df["make"].value_counts()
    rare_makes = make_counts[make_counts <= n].index
    rare_makes = df["make"].isin(rare_makes)
    df.loc[rare_makes, "make"] = "RARE"


def get_midpoint(labels: pd.Series):
    """_summary_

    Parameters
    ----------
    labels : pd.Series
        _description_

    Returns
    -------
    _type_
        _description_
    """
    labs = labels.str.replace(r"[k+$miles]", "", regex=True)
    labs = labs.str.split(" to ", n=1, expand=True)
    labs = labs.fillna(value=np.nan)
    labs[0] = labs[0].astype(float)
    labs[1] = labs[1].astype(float)
    midpoints = np.mean(labs, axis=1)
    midpoints = midpoints.fillna(labs[0] + np.std(labs[0]))
    return midpoints


def to_ordinal(data: pd.DataFrame, col: str):
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    col : str
        _description_
    """
    data[f"{col}_midpoint"] = get_midpoint(data[col])
    levels = data[[col, f"{col}_midpoint"]].sort_values(by=f"{col}_midpoint")
    levels = levels[col].unique()
    data[col] = pd.Categorical(data[col], categories=levels, ordered=True)


def fix_dtypes(data: pd.DataFrame):
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    """
    try:
        data["engine"] = data["engine"].str.strip("L")
        data["engine"] = data["engine"].astype(float)
    except AttributeError:
        pass
    try:
        data["engine_appraisal"] = data["engine_appraisal"].str.strip("L")
        data["engine_appraisal"] = data["engine_appraisal"].astype(float)
    except AttributeError:
        pass

    to_ordinal(data, "appraisal_offer")
    to_ordinal(data, "price")
    to_ordinal(data, "mileage")
    to_ordinal(data, "mileage_appraisal")
