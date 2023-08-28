
# Wrangling/math/stats
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, f1_score, accuracy_score
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
# System utils
from typing import Callable, Tuple, Optional, Dict


def accuracy_mosaic_plot(
        true_label: np.ndarray,
        predicted: np.ndarray,
        var: str,
        metric_str: str,
        overall_score: float
        ):
    """_summary_

    Parameters
    ----------
    true_label : np.ndarray
        _description_
    predicted : np.ndarray
        _description_
    var : str
        _description_
    metric_str : str
        _description_
    overall_score : float
        _description_

    Returns
    -------
    _type_
        _description_
    """
    correct = true_label == predicted
    data = pd.DataFrame({var: true_label, "Correctly Predicted": correct})
    # Prepare the data for the mosaic plot
    cross_table = pd.crosstab(data[var], data["Correctly Predicted"])
    print(cross_table)
    plot_data = cross_table.stack()

    # Create the mosaic plot
    N_x = len(data[var].unique())
    _, ax = plt.subplots(figsize=(N_x/1.5, 6))

    def props(key: str) -> Dict[str, str]:
        if "True" in key:
            return {"color": "green"}
        else:
            return {"color": "red"}

    title = f"{var}\n(Overall {metric_str} = {overall_score:.2f})"
    mosaic(plot_data, ax=ax, title=title, labelizer=lambda _: "", label_rotation=(-90,0), gap=.075, properties=props)
    ax.set_xlabel(var)
    ax.set_ylabel("Correctly Predicted")

    # Plot and close
    plt.show()
    plt.close()


def plot_predicted_vs_actual(
        actual: np.ndarray,
        predicted: np.ndarray,
        var: str,
        metric_str: str,
        overall_score: float
        ):
    """_summary_

    Parameters
    ----------
    actual : np.ndarray
        _description_
    predicted : np.ndarray
        _description_
    var : str
        _description_
    metric_str : str
        _description_
    overall_score : float
        _description_
    """
    sns.scatterplot(x=actual, y=predicted)
    plt.xlabel(f"Imputed {var}")
    plt.ylabel(f"Actual {var}")
    plt.title(f"Known {var} Values\n(Overall {metric_str} = {overall_score:.3f})")
    plt.show()
    plt.close()
        
    resids = predicted - actual
    sns.scatterplot(x=actual, y=resids)
    plt.ylabel(f"Residual ({var} predicted - actual)")
    plt.xlabel(f"Actual {var}")
    plt.title(f"{var} Imputation Model Residuals")
    plt.show()
    plt.close()


def print_scores(
        col: str,
        metric_str: str,
        train_score: float,
        test_score: float,
        previous_scores: Optional[Tuple[float, float]]
        ):
    """_summary_

    Parameters
    ----------
    col : str
        _description_
    metric_str : str
        _description_
    train_score : float
        _description_
    test_score : float
        _description_
    previous_scores : Optional[Tuple[float, float]]
        _description_
    """
    # Create basic reporting info
    train_info = f"{col} Training: {metric_str} = {train_score}"
    test_info = f"{col} Testing: {metric_str} = {test_score}"

    # Update reporting info if previous scores are present
    if previous_scores:
        prev_train = previous_scores[0]
        if train_score > prev_train:
            train_info = train_info + f" (> previous of {prev_train})"
        else:
            train_info = train_info + f" (<= previous of {prev_train})"
        
        prev_test = previous_scores[1]
        if test_score > prev_test:
            test_info = test_info + f" (> previous of {prev_test})"
        else:
            test_info = test_info + f" (<= previous of {prev_test})"

    # Print info about scores
    print(train_info)
    print(test_info)


def report_imputation_performance(
        df: pd.DataFrame,
        col: str,
        imputed_vals: np.ndarray,
        metric: Callable[..., float],
        previous_scores: Optional[Tuple[float, float]] = None,
        plot: bool = True
        ) -> Tuple[float, float]:
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    col : str
        _description_
    imputed_vals : np.ndarray
        _description_
    metric : Callable[..., float]
        _description_
    previous_scores : Optional[Tuple[float, float]], optional
        _description_, by default None
    plot : bool, optional
        _description_, by default True

    Returns
    -------
    Tuple[float, float]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # Decide how to report scores
    kwargs = {}
    if metric == r2_score:
        metric_str = "R^2"
        plt_func: Callable = plot_predicted_vs_actual
    elif metric == f1_score:
        metric_str = "micro-averaged F1 score"
        kwargs = {"average": "micro"}
        plt_func = accuracy_mosaic_plot
    elif metric == accuracy_score:
        metric_str = "classification accuracy"
        plt_func = accuracy_mosaic_plot
    else:
        raise ValueError("Pass either r2_score, f1_score, or accuracy_score from sklearn.metrics")

    # Figure out where the known/predicted data is
    if f"{col}_imputed" in df.columns:
        known = ~df[f"{col}_imputed"]
    else:
        known = ~df[col].isna()     
    known_vals = df[known][col].values
    if isinstance(imputed_vals, np.ndarray):
        predicted = imputed_vals
    else:
        predicted = imputed_vals.values

    # Calculate scores
    train_known = (df['split'] == "train") & known
    test_known = (df['split'] == "test") & known
    train_score = metric(df[train_known][col], imputed_vals[train_known], **kwargs)
    test_score = metric(df[test_known][col], imputed_vals[test_known], **kwargs)
    overall_score = metric(known_vals, predicted[known], **kwargs)

    # Generate summary figures
    if plot:
        plt_func(known_vals, predicted[known], col, metric_str, overall_score)
    
    # Print score report
    print_scores(
        col=col,
        metric_str=metric_str,
        train_score=train_score,
        test_score=test_score,
        previous_scores=previous_scores
        )

    # Return scores
    return (train_score, test_score)