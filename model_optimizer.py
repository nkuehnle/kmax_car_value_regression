import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable
import argparse
import warnings
from utils.ordinal_loss import coral_loss
from lifelines.utils.concordance import concordance_index


# List of categorical columns
CATEGORICALS = [
    "color",
    "make",
    "body",
    "trim_descrip",
    "market"
    ]

# Define the DT parameter grid for GridSearchCV
DEFAULT_DT_GRID: Dict[str, List[Any]] = {
    "criterion": ["squared_error", "poisson"],
    'max_depth': [None, 5, 10, 20, 30, 50],
    'min_samples_split': [5, 10, 15, 20, 30, 50],
    'min_samples_leaf': [2, 4, 6, 8, 15, 30, 50],
    'ccp_alpha': [0.0, 0.05, 0.1, 0.25,  0.5]
    }

# Quick version of DT parameter grid for testing environment
QUICK_TEST_DT_GRID: Dict[str, List[Any]] = {
    "criterion": ["squared_error", "poisson"],
    'max_depth': [3, 7, 15]
    }

# Define the DT parameter grid for GridSearchCV
DEFAULT_RF_GRID: Dict[str, List[Any]] = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': [None, 'sqrt', 'log2'],
    'bootstrap': [True, False],
    "criterion": ["squared_error", "poisson"],
    'max_depth': [15, 30],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 6, 8, 10],
    }

# Quick version of DT parameter grid for testing environment
QUICK_TEST_RF_GRID: Dict[str, List[Any]] = {
    "criterion": ["squared_error", "poisson"],
    'max_depth': [3, 7, 15]
    }

# Model consts
MODELS = {
    "random_forest": {
        "class": RandomForestRegressor,
        "quick_params": QUICK_TEST_RF_GRID,
        "default_params": DEFAULT_RF_GRID,
        "short_alias": "RF"
        },
    "decision_tree": {
        "class": DecisionTreeRegressor,
        "quick_params": QUICK_TEST_DT_GRID,
        "default_params": DEFAULT_DT_GRID,
        "short_alias": "DT"
        }
    }


class DataSet:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError("Features and target lengths do not match.")
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)


def rounded_scorer(y_true: np.ndarray, y_pred: np.ndarray,
                   score_func: Callable[[np.ndarray, np.ndarray], float],
                   **kwargs
                   ) -> float:
    y_pred = np.round(y_pred).astype(np.int8)
    y_true = y_true.astype(np.int8)
    return score_func(y_true, y_pred, **kwargs)


def rounded_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return rounded_scorer(y_true, y_pred, r2_score)


def rounded_coral_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return rounded_scorer(y_true, y_pred, coral_loss)


def rounded_c_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return rounded_scorer(y_true, y_pred, concordance_index)


def optimize_and_save_model(
        model: str,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        output_dir: str | Path,
        output_prefix: str,
        n_cores: int):
    """
    Optimize a DecisionTreeRegressor using GridSearchCV, pickle the best model, and save
    the search grid results to CSV.


    Parameters
    ----------
    model : str
        "random_forest" or "decision_tree" indicating what type of model to train   
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The directory to save the output files.
    param_grid : Dict[str, List[Any]]
        A dictionary of key-word arguments with associated lists of arguments to try.
    output_dir : str | Path
        _description_
    output_prefix : str
        The directory to save the output files.
    n_cores : int
        Number of CPU cores to perform GridSearch with

    """
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the decision tree regressor and scorer
    model_class: RandomForestRegressor | DecisionTreeRegressor = MODELS[model]['class']
    dt_regressor = model_class()

    # Set up custom and built-in scorers
    coral_scorer = make_scorer(
        score_func=rounded_coral_loss,
        greater_is_better=False
        )
    c_scorer = make_scorer(
        score_func=rounded_c_score,
        greater_is_better=True
        )
    r2_scorer = make_scorer(
        score_func=rounded_r2,
        greater_is_better=True
        )

    scoring_metrics = {
        "r2": r2_scorer,
        "neg_ord_cross_entropy": coral_scorer,
        "c_index": c_scorer
        }

    # Perform grid search cross-validation with progress report
    grid_search = GridSearchCV(
        estimator=dt_regressor,
        param_grid=param_grid,
        scoring=scoring_metrics,
        refit="neg_ord_cross_entropy",
        cv=5,
        n_jobs=n_cores,
        verbose=3
        )
    grid_search.fit(X, y)

    # Print the best parameters and score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Save the best model
    alias = MODELS[model]["short_alias"]
    model_filename = output_dir / f'{output_prefix}_best_{alias}.pkl'
    with open(model_filename, 'wb') as f:
        pkl.dump(best_model, f)
    print(f"Best model saved to: {model_filename}")

    # Save the grid search results to a CSV file
    results_filename = output_dir / f'{output_prefix}_{alias}_grid.csv'
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv(results_filename, index=False)
    print(f"Grid search results saved to: {results_filename}")


def _split_X_y(
        data: pd.DataFrame,
        non_features: List[str],
        target: str = "value"
        ) -> DataSet:
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    non_features : List[str]
        _description_
    target : str, optional
        _description_, by default "value"

    Returns
    -------
    DataSet
        _description_
    """
    # Check for inclusion of target in non-feature list
    if target not in non_features:
        w = f"Target {target} not included in non_features list. Automatically added."
        warnings.warn(w, DeprecationWarning)

    # Separate features and target variable
    X = data.drop(columns=non_features).copy()
    y = data[target].astype("float64")
    # Instantiate and retur nwrapper object
    dataset = DataSet(X=X, y=y)
    return dataset


def split_X_y(
        data: pd.DataFrame,
        non_features: List[str],
        target: str = "value"
        ) -> Tuple[DataSet, DataSet]:
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    non_features : List[str]
        _description_
    target : str, optional
        _description_, by default "value"

    Returns
    -------
    Tuple[DataSet, DataSet]
        _description_
    """
    training_dataset = _split_X_y(data[data["split"] == "train"], non_features, target)
    testing_dataset = _split_X_y(data[data["split"] == "test"], non_features, target)

    return training_dataset, testing_dataset


def main(
        data_path: str | Path,
        out_path: str | Path,
        quick: bool,
        n_cores: int,
        model: str
        ):
    """_summary_

    Parameters
    ----------
    data_path : str | Path
        _description_
    out_path : str | Path
        _description_
    quick : bool
        Whether to do a shallow grid search, i.e. for testing purposes
    n_cores : int
        Number of cores to pass to GridSearchCV
    model : str
        "random_forest" or "decision_tree" indicating which model to train
    """
    # Determine param grid
    if quick:
        param_grid = MODELS[model]["quick_params"]
    else:
        param_grid = MODELS[model]["default_params"]

    # Get list of files in data path
    data_path = Path(data_path)
    data_files = data_path.glob("*.pkl")  # Get .pkl files

    # Create dict mapping of model names to datasets to train on
    data_to_model: Dict[str, Tuple[DataSet, DataSet]] = {}

    for data_file in data_files:
        # Import data
        data_file = Path(data_file)
        if data_file.suffix == ".pkl":
            data: pd.DataFrame = pd.read_pickle(data_file)
        elif data_file.suffix == ".csv":
            data = pd.read_csv(data_file)

        # Get target/feature splits as wrapper object
        non_features = ["value", "split", "model"]
        non_features = non_features + [c for c in data.columns if "imputed" in c]
        training_data, testing_data = split_X_y(
            data=data,
            non_features=non_features,
            target="value"
            )

        categorical_transformer = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(), CATEGORICALS)],
            remainder='passthrough'
        )
        training_data.X = categorical_transformer.fit_transform(training_data.X)
        testing_data.X = categorical_transformer.transform(testing_data.X)

        data_to_model[data_file.stem] = (training_data, testing_data)
        
        
    # Optimize and save each variation we are testing
    for model_name, model_data in data_to_model.items():
        mod_train_dat, mod_test_dat = model_data

        test_X_out = data_path/f"{model_name}_test_X.pkl"
        if not test_X_out.is_file():
            with open(test_X_out, "wb") as tx:
                pkl.dump(mod_test_dat.X, tx)

        test_y_out = data_path/f"{model_name}_test_y.pkl"
        if not test_y_out.is_file():
            with open(test_y_out, "wb") as ty:
                pkl.dump(mod_test_dat.y, ty)

        alias = MODELS[model]["short_alias"]
        model_filename = Path(out_path) / f'{model_name}_best_{alias}.pkl'

        if not model_filename.is_file():
            optimize_and_save_model(
                model=model,
                X=mod_train_dat.X,
                y=mod_train_dat .y,
                param_grid=param_grid,
                output_dir=out_path,
                output_prefix=model_name,
                n_cores=n_cores,
                )


def parse_arguments() -> Tuple[Path, Path, int, bool, str]:
    """
    Parse command-line arguments.

    Returns
    -------
    Tuple[Path, Path]
        A pair of paths to the pickled data and the path to save outputs.
    """
    parser = argparse.ArgumentParser(
        description='Perform GridSearchCV for a decision tree.'
        )
    # Input/output locations
    parser.add_argument(
        '-d',
        '--data',
        type=str,
        help='Path to data')
    parser.add_argument(
        '-o',
        '--outpath',
        type=str,
        help='Output path'
        )
    # Model type
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--dt',
        action='store_const',
        const='decision_tree',
        dest="model",
        help="Optimize Decision Tree"
        )
    group.add_argument(
        '--rf',
        action='store_const',
        const='random_forest',
        dest="model",
        help="Optimize Random Forest"
        )
    # Speed/ other settings
    parser.add_argument(
        '-p',
        '--cores',
        default=-1,
        type=int,
        help="Number of CPU cores"
        )
    parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        default=False,
        help="Enable quick test mode"
        )
    # Get args and return
    args = parser.parse_args()
    return Path(args.data), Path(args.outpath), args.cores, args.quick, args.model


if __name__ == "__main__":
    data_path, out_path, n_cores, quick, model = parse_arguments()
    main(
        data_path=data_path,
        out_path=out_path,
        quick=quick,
        n_cores=n_cores,
        model=model
        )
