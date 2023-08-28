import numpy as np
from functools import wraps
from lifelines.utils.concordance import concordance_index

def chunked_difference(array: np.ndarray, n_chunks: int = 1) -> np.ndarray:
    if n_chunks <= 0:
        raise ValueError("n must be a positive integer.")
    
    chunks = np.array_split(array, n_chunks)
    diffs = []
    
    for i, chunk in enumerate(chunks):
        self_diff = np.subtract.outer(chunk, chunk)
        self_diff = self_diff[np.triu_indices(self_diff.shape[0], k=1)]
        diffs.append(self_diff)
        for ochunk in chunks[i+1:]:
            other_diff = np.subtract.outer(chunk, ochunk).flatten()
            diffs.append(other_diff)
        
    return np.concatenate(diffs)


def get_differences(array: np.ndarray, n_chunks: int = 1) -> np.ndarray:
    if n_chunks == 1:
        diffs = np.subtract.outer(array, array)
        diffs = diffs[np.triu_indices(diffs.shape[0], k=1)]
        return diffs
    else:
        return chunked_difference(array, n_chunks)
    

def rank_correlation(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        flavor: str = "c_index",
        n_comp_chunks: int = 1
        ) -> float:
    """
    Calculates the rank correlation given true ranks and predicted ranks using the
    ratio of concordant pairs to total possible pairs.
    
    For pair i,j the pair is concordant for true ranks T and predicted ranks T if they
    differ in the same direction, i.e. true tanks T_i > T_J and predicted ranks
    P_i > P_j (or T_i < T_j and P_i < P_j). Discordant pairs are those where the
    opposite is true, i.e. T_i > T_j and P_i < T_j.
    
    Ties (in the predictor) can either be ignored altogether or given a partial weight
    (i.e. .5), ties in the true/observed values are treated as incomparable.
    
    A higher correlation is always indicative of better predictive performance.

    Three flavors are provided with different handling of ties.
    - If ties are ignored, this reduces to the Goodman-Kruskal γ statistic.
        - Treats all ties as incomparable
        - Ranges from -1 (anti-correlated) to 1 (perfectly correlated), 0 is expected of
          random behavior
    - If ties (in the predictions) are considered but weighted at 0, it is Sommar's D.
        - Treats ties as discordant pairs
        - Ranges from -1 (anti-correlated) to 1 (perfectly correlated), 0 is expected of
        random behavior
    - If ties (in the predictions) are considered and weighted by .5 it is Harrell's C.
        - Treats ties in between concordant and discordant
        - Ranges from 0 (anti-correlated/anti-concordant) to 1 (perfectly 
          correlated/concordant), .5 is expected of random behavior

    See:
    Therneau T and Atkinson E.
    Concordance. CRAN March 11, 2023.
    https://cran.r-project.org/web/packages/survival/vignettes/concordance.pdf          


    Harrell FE, Califf RM, Pryor DB, Lee KL, Rosati RA.
    Evaluating the Yield of Medical Tests. JAMA. 1982;247(18):254-2546.
    doi:10.1001/jama.1982.03320430047030


    Parameters
    ----------
    y_true : np.ndarray
        Array of true ranks.
    y_pred : np.ndarray
       Array of predicted ranks.
    flavor : str
        A string indicating the type of rank correlation to calculate,
        by default "c_index"

    Returns
    -------
    float
        Calculated rank correlation of data

    Raises
    ------
    ValueError
        Raised if predicted and true ranks are not of equal size.
    """
    # Ensure the input arrays are NumPy arrays
    valid_flavors = ("goodman_krusal", "somers_d", "c_index")
    if flavor not in valid_flavors:
        vflavors = ", ".join([f"'{i}'" for i in valid_flavors])
        raise ValueError(f"Invalid flavor {flavor}: select from {vflavors}")

    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of y_true and y_pred must be the same.")
    
    diff_true = get_differences(y_true, n_chunks=n_comp_chunks)
    true_tied = diff_true == 0
    diff_true = diff_true[~true_tied]

    diff_pred = get_differences(y_pred, n_chunks=n_comp_chunks)
    diff_pred = diff_pred[~true_tied]
    del true_tied

    # Count valid pairs
    true_greater = diff_true > 0
    true_lesser = diff_true < 0
    del diff_true
    pred_greater = diff_pred > 0
    pred_lesser = diff_pred < 0
    pred_tied = diff_pred == 0
    del diff_pred

    # Total up different pairs
    concord_mask = (true_greater & pred_greater) | (true_lesser & pred_lesser)
    concord_pairs: int = np.sum(concord_mask)
    discord_mask = (true_greater & pred_lesser) | (true_lesser & pred_greater)
    discord_pairs: int = np.sum(discord_mask)
    del true_greater
    del true_lesser
    del pred_greater
    del pred_lesser
    tied_pairs: int = np.sum(pred_tied)

    tie_mask = 1 if flavor in ("somers_d", "c_index") else 0
    total_pairs = concord_pairs + discord_pairs + (tie_mask * tied_pairs)

    rank_correlation = (concord_pairs - discord_pairs) / total_pairs

    if flavor == "c_index":
        return (rank_correlation + 1)/2
    else:
        return rank_correlation


# Convenience functions
@wraps(rank_correlation)
def harrells_c(y_true: np.ndarray, y_pred: np.ndarray, n_comp_chunks: int) -> float:
    return rank_correlation(y_true, y_pred, "c_index", n_comp_chunks)


@wraps(rank_correlation)
def somers_d(y_true: np.ndarray, y_pred: np.ndarray, n_comp_chunks: int) -> float:
    return rank_correlation(y_true, y_pred, "somers_d", n_comp_chunks)


@wraps(rank_correlation)
def goodman_kruskals(y_true: np.ndarray, y_pred: np.ndarray, n_comp_chunks: int) -> float:
    return rank_correlation(y_true, y_pred, "somers_d", n_comp_chunks)

t = np.random.choice(range(1, 11), size=4000).astype(np.int16)
p = np.random.choice(range(1, 11), size=4000).astype(np.int16)
