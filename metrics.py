if True:
    import sys
    sys.path.insert(0, "../my_autosklearn")
    from autosklearn.metrics import roc_auc, make_scorer


from scipy.stats import pearsonr, t
from math import sqrt, log
from functools import partial
import numpy as np
from sklearn.metrics import (
    recall_score, 
    precision_score, 
    roc_curve,
    auc
)


def mean(it):
    t = [e for e in it if e is not None]
    n = len(t)
    if n == 0:
        return None
    return sum(t) / n


def med(it):
    t = sorted(e for e in it if e is not None)
    n = len(t)
    if n == 0:
        return None
    return (t[(n-1)//2] + t[n//2]) / 2

    
def imbalance_ratio(y, mode):
    (l1, l2), (c1, c2) = np.unique(y, return_counts=True)
    (c1, l1), (c2, l2) = sorted(((c1, l1), (c2, l2)))
    if mode == "small":
        return c1 / c2
    if mode == "big":
        return c2 / c1
    assert False, f"Invalid mode: {mode}."


def number_of_discriminative_features(X, y):
    N = len(y)
    alpha = 0.05

    p_star = 0
    for col in X.columns:
        xj = X[col].to_numpy()
        rj = pearsonr(xj, y).pvalue
        tj = rj * sqrt((N-2) / (1-rj**2))
        p_star += not (-t.ppf(1-alpha, N-2) < tj < t.ppf(1-alpha, N-2))
    return p_star


def adjusted_imbalance_ratio(X, y, p_star=None):
    IR = imbalance_ratio(y, mode="big")
    lambda_ = 1
    if p_star is None:
        p_star = number_of_discriminative_features(X, y)

    return IR - lambda_ * log(1 if p_star == 0 else p_star)


def partial_roc_auc_score_(y_true, y_prob):
    def point_on_line(x1, y1, x2, y2, x3):
        """compute y-coordinate of a point [x3, y3] that lies on 
        line between points [x1, y1] and [x2, y2]"""
        if x1 == x2:
            return y1

        dx = x2 - x1
        dy = y2 - y1
        return y1 + dy * (x3-x1) / dx

    def partial_roc_curve(y_true, y_prob, IR):
        fpr, tpr, ths = roc_curve(y_true, y_prob)
        if 1 <= IR:
            return fpr, tpr, ths

        partial_fpr = fpr[fpr < IR]
        n = len(partial_fpr)
        partial_fpr = np.append(partial_fpr, IR)
        x3 = IR
        x1, y1 = fpr[n-1], tpr[n-1]
        x2, y2 = fpr[n], tpr[n]
        y3 = point_on_line(x1, y1, x2, y2, x3)
        partial_tpr = np.append(tpr[:n], y3)
        partial_ths = ths[:n+1]
        return partial_fpr, partial_tpr, partial_ths

    # NOTE: assuming y_true follows original distribution
    IR = imbalance_ratio(y_true, mode="small")
    fpr, tpr, _ = partial_roc_curve(y_true, y_prob, IR)
    return auc(fpr, tpr) / IR


def harmonic_mean_recall_(y_true, y_pred, weight=1):
    minority_recall = recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0.0)
    majority_recall = recall_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0.0)
    return (weight+1)*minority_recall*majority_recall/(minority_recall+weight*majority_recall)


NEW_ASKL_METRICS = {
    "minority_precision": make_scorer(
        name="minority_precision",
        score_func=partial(precision_score, pos_label=1, average='binary', zero_division=0.0),
        optimum=1,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
    ),
    "minority_recall": make_scorer(
        name="minority_recall",
        score_func=partial(recall_score, pos_label=1, average='binary', zero_division=0.0),
        optimum=1,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
    ),
    "majority_recall": make_scorer(
        name="majority_recall",
        score_func=partial(recall_score, pos_label=0, average='binary', zero_division=0.0),
        optimum=1,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
    ),
    "roc_auc": roc_auc,
    "partial_roc_auc_score": make_scorer(
        name="partial_roc_auc_score",
        score_func=partial_roc_auc_score_,
        optimum=1,
        greater_is_better=True,
        needs_threshold=True,
    ),
    "harmonic_mean_recall": make_scorer(
        name="harmonic_mean_recall",
        score_func=harmonic_mean_recall_,
        optimum=1,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
    ),
    "harmonic_mean_recall_2": make_scorer(
        name="harmonic_mean_recall_2",
        score_func=partial(harmonic_mean_recall_, weight=2),
        optimum=1,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
    )
}