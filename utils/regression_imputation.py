# Wrangling/math/stats
import pandas as pd
import numpy as np
from scipy.sparse import _csr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
# Basic utils/typehinting
from typing import Callable, Tuple, List, Dict, Any, Optional
import warnings
# Custom modules
from .preprocessing import get_midpoint
from .imputation_performance import report_imputation_performance

CATEGORICALS = [
    "color",
    "make",
    "body",
    "market",
    "itransaction"
    ]


def fix_trim(imputed_vals: np.ndarray) -> np.ndarray:
    new_vals = np.where(imputed_vals, "Premium", "Not Premium")
    return new_vals


def get_model_data(data: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    model_data = data.copy()
    model_data = model_data.drop(columns=["model"])
    model_data["mileage"] = get_midpoint(model_data["mileage"])
    model_data["trim_descrip"] = model_data["trim_descrip"] == "Premium"
    
    return model_data


def get_untrained_model(
        model_class: LinearRegression | LogisticRegression,
        model_kwargs: Dict[str, Any]
        ) -> LinearRegression | LogisticRegression:
    """_summary_

    Parameters
    ----------
    model_class : LinearRegression | LogisticRegression
        _description_
    model_kwargs : Dict[str, Any]
        _description_

    Returns
    -------
    LinearRegression | LogisticRegression
        _description_
    """
    model: LinearRegression | LogisticRegression = model_class(
        n_jobs=-1,
        **model_kwargs
        )
    
    return model


def find_missing_cats(
        model_data: pd.DataFrame,
        target_col: str
        ) -> Dict[str, List[str]]:
    train_data = model_data[model_data["split"] == "train"]
    imputed_train_rows = train_data[f"{target_col}_imputed"]

    missing_cats = {}

    for cat_var in CATEGORICALS:
        all_cats = set(model_data[cat_var].unique())
        known_cats = set(train_data[~imputed_train_rows][cat_var].unique())
        if all_cats == known_cats:
            _missing_cats = []
        elif len(known_cats) < len(all_cats):
            w = f"Variable {cat_var} has categories for which {target_col} is entirely NAs in training split"
            warnings.warn(w, UserWarning)
            _missing_cats = [c for c in all_cats if c not in known_cats]
        else:
            raise ValueError("Training data has categories not present in full dataset!")
        
        missing_cats[cat_var] = _missing_cats

    missing_cats = {k: v for k, v in missing_cats.items() if any(v)}

    return missing_cats
  

def get_trainX_dataX_trainY_Yencoder(
        model_data: pd.DataFrame,
        target_col: str,
        also_exclude: List[str] = [],
        cats_to_merge_in_training: Dict[str, List[str]] = {}
        ) -> Tuple[_csr.csr_matrix, _csr.csr_matrix, np.ndarray,  np.ndarray]:
    """_summary_

    Parameters
    ----------
    model_data : pd.DataFrame
        _description_
    target_col : str
        _description_
    also_exclude : List[str], optional
        _description_, by default []
    cats_to_merge_in_training : Dict[str, List[str]], optional
        _description_, by default {}

    Returns
    -------
    Tuple[_csr.csr_matrix, _csr.csr_matrix, np.ndarray,  np.ndarray]
        _description_
    """
    # Subset on global training data
    train_data = model_data[model_data['split'] == "train"].copy()

    # Figure out what columns are needed for prediction and instantiate transformer
    exclude = ["value", "split", target_col] + also_exclude
    predictors = [c for c in train_data.columns if c not in exclude]
    predictors = [p for p in predictors if "imputed" not in p]
    categorical_predictors = [c for c in CATEGORICALS if c in predictors]
    categorical_transformer = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), categorical_predictors)],
        remainder='passthrough'
    )
    progress = f"Modeling with the following predictors: {', '.join(predictors)}"
    print(progress)
    
    # Get training df for imputation specifically
    known = ~train_data[f"{target_col}_imputed"]
    included_rows = [known]
    for var, cats in cats_to_merge_in_training.items():
        to_add = train_data[var].isin(cats)
        included_rows.append(to_add)

    # Note as of July 2023 Flake8 doesn't understand this.
    training_mask: np.ndarray = np.logical_or.reduce(included_rows)

    train = train_data[training_mask]

    # Get column-transformed matrices for training and imputation
    train_X = categorical_transformer.fit_transform(train[predictors])
    data_X = categorical_transformer.transform(model_data[predictors])
    train_Y = train[target_col].values

    if target_col in CATEGORICALS:
        Y_encoder = LabelEncoder()
        train_Y = Y_encoder.fit_transform(train_Y)
    else:
        Y_encoder = False

    return (train_X, data_X, train_Y, Y_encoder)


def impute_vals(
        model: LinearRegression | LogisticRegression,
        train_X: _csr.csr_matrix,
        data_X: _csr.csr_matrix,
        train_Y: np.ndarray,
        encoder: Optional[OneHotEncoder] = None
        ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    model : LinearRegression | LogisticRegression
        _description_
    train_X : _csr.csr_matrix
        _description_
    data_X : _csr.csr_matrix
        _description_
    train_Y : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    # Fit model and impute values
    model.fit(train_X, train_Y)
    imputed_vals = model.predict(data_X)

    if encoder:
        imputed_vals = encoder.inverse_transform(imputed_vals)

    return imputed_vals


def iteratively_impute_vals(
        model_data: pd.DataFrame,
        target_col: str,
        missing_cats: Dict[str, List[str]],
        imputed_vals: np.ndarray,
        model_class: LinearRegression | LogisticRegression,
        model_kwargs: Dict[str, Any]
        ):
    """_summary_

    Parameters
    ----------
    model_data : pd.DataFrame
        _description_
    target_col : str
        _description_
    missing_cats : Dict[str, List[str]]
        _description_
    imputed_vals : np.ndarray
        _description_
    model_class : LinearRegression | LogisticRegression
        _description_
    model_kwargs : Dict[str, Any]
        _description_

    Returns
    -------
    _type_
        _description_
    """
    _model_data = model_data.copy()
    
    for var, cats in missing_cats.items():
        missing_inds = model_data[var].isin(cats)
        _model_data.loc[missing_inds, target_col] = imputed_vals[missing_inds]

    # Get training data and prediction data
    train_X, data_X, train_Y, Y_encoder = get_trainX_dataX_trainY_Yencoder(
        model_data=model_data,
        target_col=target_col,
        cats_to_merge_in_training=missing_cats
        )
    
    # Get model (untrained)
    model = get_untrained_model(model_class, model_kwargs)

    # Get imputed values
    new_imputed_vals = impute_vals(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        data_X=data_X,
        encoder=Y_encoder
        )
    
    return new_imputed_vals


def impute_by_regression(
        data: pd.DataFrame,
        target_col: str,
        model_class: LinearRegression | LogisticRegression,
        metric: Callable[..., float],
        model_kwargs: Dict[str, Any] = {},
        report_kwargs: Dict[str, Any] = {}
        ):
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    target_col : str
        _description_
    model_class : LinearRegression | LogisticRegression
        _description_
    metric : Callable[..., float]
        _description_
    model_kwargs : Dict[str, Any], optional
        _description_, by default {}
    report_kwargs : Dict[str, Any], optional
        _description_, by default {}
    """
    model_data = get_model_data(data=data)

    # Figure out if any categorical data lies exclusively within the missing data here
    missing_cats = find_missing_cats(model_data=model_data, target_col=target_col)
    also_exclude = [k for k in missing_cats.keys()]

    # Update progress
    progress = "initial" if any(missing_cats) else ""
    progress = f"Fitting {progress} regression model to impute missing data."
    print(progress)

    # Get training data and prediction data
    train_X, data_X, train_Y, Y_encoder = get_trainX_dataX_trainY_Yencoder(
        model_data=model_data,
        target_col=target_col,
        also_exclude=also_exclude
        )

    # Create untrained model
    model = get_untrained_model(model_class, model_kwargs)

    # Get imputed values    
    imputed_vals1 = impute_vals(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        data_X=data_X,
        encoder=Y_encoder
        )

    # Update progress
    if any(also_exclude):
        exclude_str = ', '.join(also_exclude)
        progress = f"\nInitially excluding: {exclude_str}"
        print(progress)

    imputed_vals = fix_trim(imputed_vals1) if "trim" in target_col else imputed_vals1

    _ = report_imputation_performance(
        df=data,
        col=target_col,
        imputed_vals=imputed_vals,
        metric=metric,
        **report_kwargs)

    if any(missing_cats):
        # Update progress
        bad_cats = [", ".join(cats) for cats in missing_cats.values()]
        problems = [f"{var} ({cats})" for var, cats in zip(also_exclude, bad_cats)]
        progress = "; ".join(problems)
        progress = "Performing second pass using imputed values for: " + progress
        print(progress)

        # Get second set of values
        imputed_vals2 = iteratively_impute_vals(
            model_data=model_data,
            target_col=target_col,
            missing_cats=missing_cats,
            imputed_vals=imputed_vals1,
            model_class=model_class,
            model_kwargs=model_kwargs
            )

        imputed_vals = fix_trim(imputed_vals2) if "trim" in target_col else imputed_vals2

        _ = report_imputation_performance(
            df=data,
            col=target_col,
            imputed_vals=imputed_vals,
            metric=metric,
            **report_kwargs
            )
        
    return imputed_vals