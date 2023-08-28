import numpy as np
from sklearn.metrics import log_loss

def coral_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        importance_weights: np.ndarray | str = "auto"
        ) -> float:
    """
    Compute the CORAL (consistent rank logits) Loss for ordinal regression.

    See:
    Wenzhi Cao, Vahid Mirjalili, Sebastian Raschka,
    Rank consistent ordinal regression for neural networks with application to age estimation,
    Pattern Recognition Letters,
    Volume 140,
    2020,
    Pages 325-331,
    ISSN 0167-8655,

    CORAL is the weighted sum of K-1 binary log loss problems. Here K is the max rank.

    Parameters
    ----------
    y_true : np.ndarray
        True ordinal labels.
    y_pred : np.ndarray
        Predicted ordinal labels.
    importance_weights : np.ndarray | str, optional
        Weight for the K-1 binary classier, by default "auto"
        If "auto," weights are based on the true abundance of each binary task within
        the amongst the true labels.

    Returns
    -------
    float
        Cumulative Logit Loss value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get array of possible ranks
    possible_ranks = np.concatenate((y_true, y_pred))
    possible_ranks = np.arange(possible_ranks.max())

    binarized_true = np.less_equal.outer(y_true, possible_ranks)
    binarized_pred = np.less_equal.outer(y_pred, possible_ranks)

    # Get K
    K = possible_ranks.shape[0]

    if not isinstance(importance_weights, str):
        iweights: np.ndarray = importance_weights
    else:
        # Set importance by relative abundance amongst true cases
        iweights = binarized_true.sum(axis=0)/binarized_true.sum()

    log_losses = np.zeros(K)

    for i in range(K-1):
        log_losses[i] += log_loss(
            y_true=binarized_true[:, i],
            y_pred=binarized_pred[:, i],
            labels=[True, False]
            )

    weighted_log_loss = log_losses * iweights
    ordinal_loss = weighted_log_loss.sum()

    return ordinal_loss
