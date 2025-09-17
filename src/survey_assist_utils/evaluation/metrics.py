"""Calculation of simple evaluation metrics."""

import pandas as pd
from pydantic import BaseModel

from survey_assist_utils.evaluation.code_comparison import compare_codes


class AmbiguityMetrics(BaseModel):
    """Metrics for ambiguity detection: precision, recall, F1-score,
    and confusion matrix counts.
    """

    precision: float
    recall: float
    f1: float
    TP: int
    FP: int
    FN: int
    TN: int

    def print_metrics(self):
        """Pretty print the ambiguity detection metrics."""
        print("\nAmbiguity decision statistics:")
        print(f"Precision: {100 * self.precision:.2f}%")
        print(f"Recall: {100 * self.recall:.2f}%")
        print(f"F1-score: {100 * self.f1:.2f}%")
        print(f"TP: {self.TP}, FP: {self.FP}, FN: {self.FN}, TN: {self.TN}")


class AccuracyMetrics(BaseModel):
    """Metrics for accuracy evaluation: total records, unambiguous records,
    matches (MM and OO), and accuracy (MM and OO).
    """

    total_records: int
    unambiguous_records: int
    matches_mm: int
    accuracy_mm_total: float
    matches_oo: int
    accuracy_oo_total: float
    accuracy_oo_unambiguous: float

    def print_metrics(self, title: str = "Initial"):
        """Pretty print the accuracy metrics."""
        print(f"\n{title} classification accuracy metrics:")
        print(f"{title} accuracy (OO, full set): {100 * self.accuracy_oo_total:.2f}%")
        print(
            f"{title} accuracy (OO, unambiguous subset by CC&SA): "
            f"{100 * self.accuracy_oo_unambiguous:.2f}%"
        )
        print(f"{title} accuracy (MM, full set): {100 * self.accuracy_mm_total:.2f}%")


class CodabilityMetrics(BaseModel):
    """Metrics for codability: initial and final codable proportions,
    improvement in codability, and counts of codable records.
    """

    initial_codable_prop: float
    final_codable_prop: float
    codability_improvement_prop: float
    initial_codable_count: int
    final_codable_count: int

    def print_metrics(self):
        """Pretty print the codability metrics."""
        print("\nCodability metrics:")
        print(f"Initial codability: {100 * self.initial_codable_prop:.2f}%")
        print(f"Final codability: {100 * self.final_codable_prop:.2f}%")
        print(f"Gain in codability: {100 * self.codability_improvement_prop:.2f}pp")


class SimpleMetrics(BaseModel):
    """Container for all simple evaluation metrics."""

    ambiguity_metrics: AmbiguityMetrics
    codability_metrics: CodabilityMetrics
    initial_accuracy_metrics: AccuracyMetrics
    final_accuracy_metrics: AccuracyMetrics

    def print_metrics(self):
        """Pretty print all simple metrics."""
        self.ambiguity_metrics.print_metrics()
        self.codability_metrics.print_metrics()
        self.initial_accuracy_metrics.print_metrics("Initial")
        self.final_accuracy_metrics.print_metrics("Final")


def calc_ambiguity_metrics(
    df: pd.DataFrame,
    model_col: str = "sa_initial_ambiguous",
    truth_col: str = "clerical_ambiguous",
) -> AmbiguityMetrics:
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

    return AmbiguityMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        TP=true_pos,
        FP=false_pos,
        FN=false_neg,
        TN=true_neg,
    )


def calc_accuracy_metrics(
    df: pd.DataFrame,
    model_col: str = "sa_initial_codes",
    truth_col: str = "clerical_codes",
) -> AccuracyMetrics:
    """Calculate classification accuracy metrics.

    Args:
        df: DataFrame containing model and clerical code columns.
        model_col: Column name for model predicted codes (string or list/set).
        truth_col: Column name for true (clerical) codes (string or list/set).

    Returns:
        Dictionary with accuracy and counts of matches/non-matches.
    """
    total = len(df)
    if total == 0:
        return AccuracyMetrics(
            total_records=0,
            unambiguous_records=0,
            matches_mm=0,
            accuracy_mm_total=0.0,
            matches_oo=0,
            accuracy_oo_total=0.0,
            accuracy_oo_unambiguous=0.0,
        )

    unambiguous = sum((df[truth_col].apply(len) == 1) & (df[model_col].apply(len) == 1))

    matches = {}

    def compare_row(row: pd.Series, method) -> bool:
        return compare_codes(row[truth_col], row[model_col], method=method)

    for method in ["OO", "MM"]:
        df["match"] = df.apply(compare_row, method=method, axis=1)
        matches[method] = df["match"].sum()

    accuracy_oo = matches["OO"] / total
    accuracy_mm = matches["MM"] / total
    accuracy_oo_unambiguous = matches["OO"] / unambiguous if unambiguous > 0 else 0.0

    return AccuracyMetrics(
        total_records=total,
        unambiguous_records=unambiguous,
        matches_mm=matches["MM"],
        accuracy_mm_total=accuracy_mm,
        matches_oo=matches["OO"],
        accuracy_oo_total=accuracy_oo,
        accuracy_oo_unambiguous=accuracy_oo_unambiguous,
    )


def calc_simple_metrics(
    df: pd.DataFrame,
    truth_col: str = "clerical_codes",
    initial_model_col: str = "sa_initial_codes",
    final_model_col: str = "sa_final_codes",
) -> SimpleMetrics:
    """Calculate ambiguity detection and classification accuracy metrics.

    Args:
        df: DataFrame containing model and clerical code columns.
        truth_col: Column name for true (clerical) codes (string or list/set).
        initial_model_col: Column name for initial model predicted codes.
        final_model_col: Column name for final model predicted codes.

    Returns:
        Dictionary with calculated metrics.
    """
    df = df[[initial_model_col, final_model_col, truth_col]].copy()
    df["truth_ambiguous"] = df[truth_col].apply(lambda x: len(x) != 1)
    df["initial_ambiguous"] = df[initial_model_col].apply(lambda x: len(x) != 1)
    df["final_ambiguous"] = df[final_model_col].apply(lambda x: len(x) != 1)

    # Calculate ambiguity metrics
    ambig_metrics = calc_ambiguity_metrics(
        df, model_col="initial_ambiguous", truth_col="truth_ambiguous"
    )

    total_count = len(df)
    initial_codable = sum(~df["initial_ambiguous"])
    final_codable = sum(~df["final_ambiguous"])
    codability_metrics = CodabilityMetrics(
        initial_codable_prop=initial_codable / total_count if total_count > 0 else 0.0,
        final_codable_prop=final_codable / total_count if total_count > 0 else 0.0,
        codability_improvement_prop=(
            (final_codable - initial_codable) / total_count if total_count > 0 else 0.0
        ),
        initial_codable_count=initial_codable,
        final_codable_count=final_codable,
    )

    # Calculate classification accuracy metrics
    initial_accuracy_metrics = calc_accuracy_metrics(
        df, model_col=initial_model_col, truth_col=truth_col
    )

    final_accuracy_metrics = calc_accuracy_metrics(
        df,
        model_col="sa_final_codes",
        truth_col="clerical_codes",
    )

    return SimpleMetrics(
        ambiguity_metrics=ambig_metrics,
        codability_metrics=codability_metrics,
        initial_accuracy_metrics=initial_accuracy_metrics,
        final_accuracy_metrics=final_accuracy_metrics,
    )
