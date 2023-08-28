import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
from typing import Dict, Optional
import pickle as pkl

# Custom modules
from .consts import LONG_MAPPER, SHORT_MAPPER, SHARED_COLS
from .preprocessing import (
    to_ordinal,
    fix_dtypes,
)

def load_carmax_data(data_path: Path) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    data_path : Path
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    data = pd.read_csv(data_path)
    fix_dtypes(data)

    # Fix column names/set final columns
    mapper = {c: f"{c}_purchase" for c in data.columns if "appraisal" not in c}
    mapper["market"] = "market"
    mapper["price"] = "value_purchase"
    mapper["appraisal_offer"] = "value_appraisal"
    
    data = data.rename(columns=mapper)
    data = data[[c for c in data.columns if "midpoint" not in c]]

    for int_col in ["model_year", "cylinders"]:
        data[f"{int_col}_appraisal"] = data[f"{int_col}_appraisal"].astype("Int64")
        data[f"{int_col}_purchase"] = data[f"{int_col}_purchase"].astype("Int64")

    return data


def carmax_short_to_long(
    kmax_data: pd.DataFrame, additional_cols: Dict[str, str] = None
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    kmax_data : pd.DataFrame
        _description_
    additional_cols : Dict[str, str], optional
        _description_, by default None

    Returns
    -------
    pd.DataFrame
        _description_
    """

    # Get column name mappings
    mapper = LONG_MAPPER
    if isinstance(additional_cols, dict):
        mapper.update(additional_cols)
    mapper = {k: v for k, v in mapper.items() if k in kmax_data.columns}

    purchase_cols = [i for i in mapper.values() if i in kmax_data.columns]
    appraisal_cols = [i for i in mapper.keys() if i in kmax_data.columns]

    # Split purchase and appraisal columns into separate dataframes
    purchase_data = kmax_data[purchase_cols].copy().reset_index(drop=True)
    appraisal_data = kmax_data[appraisal_cols].copy().reset_index(drop=True)

    # Add indicator for purchase vs appraisal
    purchase_data["itransaction"] = "purchase"
    appraisal_data["itransaction"] = "appraisal"

    # Re-index appraisal data & rename coluumns
    appraisal_data.index = appraisal_data.index + len(purchase_data)
    appraisal_data = appraisal_data.rename(columns=mapper)

    # Merge into long dataframe
    long_data = pd.concat([purchase_data, appraisal_data], axis=0, ignore_index=True)
    long_data = long_data.rename(
        columns={c: c.replace("_purchase", "") for c in long_data.columns}
    )

    return long_data


def carmax_long_to_short(
    kmax_data: pd.DataFrame, additional_cols: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    kmax_data : pd.DataFrame
        _description_
    additional_cols : Dict[str, str], optional
        _description_, by default None

    Returns
    -------
    pd.DataFrame
        _description_
    """

    mapper = SHORT_MAPPER
    if isinstance(additional_cols, dict):
        mapper.update(additional_cols)
    mapper = {k: v for k, v in mapper.items() if k in kmax_data.columns}

    purchase_data = kmax_data[kmax_data["itransaction"] == "purchase"].copy()
    purchase_data = purchase_data[[v for v in mapper.keys()]].reset_index(drop=True)
    for c in purchase_data.columns:
        if (c not in SHARED_COLS) and ("purchase" not in c):
            purchase_data = purchase_data.rename(columns={c: f"{c}_purchase"})

    appraisal_data = kmax_data[kmax_data["itransaction"] == "appraisal"].copy()
    appraisal_data = appraisal_data.rename(columns=mapper)
    appraisal_data = appraisal_data[[i for i in mapper.values()]].reset_index(drop=True)
    
    for c in ['value', 'mileage']:
        if not is_numeric_dtype(kmax_data[c]):
            to_ordinal(kmax_data, c)

    new_data = pd.concat([appraisal_data, purchase_data], axis=1)

    return new_data


def save_sklearn_model(model: object, path: Path):
    """_summary_

    Parameters
    ----------
    model : object
        _description_
    path : Path
        _description_
    """
    with open(path, "wb") as f:
        pkl.dump(model, f, protocol=pkl.HIGHEST_PROTOCOL)
