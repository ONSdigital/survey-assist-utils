"""Calculation of simple evaluation metrics."""

import pandas as pd


def calc_ambiguity_metrics(
    df: pd.DataFrame,
    model_col: str = "sa_initial_ambiguous",
    truth_col: str = "clerical_ambiguous",
) -> dict:
    """Calculate ambiguity detection metrics: precision, recall, F1-score.

    Args:
        df: DataFrame containing model and clerical ambiguity columns.
        model_col: Column name for model ambiguity predictions (boolean).
        truth_col: Column name for true (clerical) ambiguity labels (boolean).

    Returns:
        Dictionary with precision, recall, and F1-score.
    """
    true_pos = sum(df[model_col] & df[truth_col])
    false_pos = sum(df[model_col] & ~df[truth_col])
    false_neg = sum(~df[model_col] & df[truth_col])
    true_neg = sum(~df[model_col] & ~df[truth_col])

    precision = 0 if true_pos + false_pos == 0 else true_pos / (true_pos + false_pos)
    recall = 0 if true_pos + false_neg == 0 else true_pos / (true_pos + false_neg)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * (precision * recall) / (precision + recall)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": true_pos,
        "FP": false_pos,
        "FN": false_neg,
        "TN": true_neg,
    }
