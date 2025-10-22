"""Calculation of simple evaluation metrics."""

import logging

import pandas as pd
from pydantic import BaseModel

from survey_assist_utils.evaluation.code_comparison import (
    _compare_codes,
    cast_code_to_set,
)

logger = logging.getLogger(__name__)


class AmbiguityMetrics(BaseModel):
    """Metrics for ambiguity detection: precision, recall, F1-score,
    and confusion matrix counts.
    """

    precision: float
    recall: float
    f1: float
    accuracy: float
    TP: int
    FP: int
    FN: int
    TN: int

    def report_metrics(self):
        """Pretty print the ambiguity detection metrics."""
        lines = [
            "\nAmbiguity decision statistics:",
            f" F1-score: {100 * self.f1:.2f}%",
            f" Precision: {100 * self.precision:.2f}%",
            f" Recall: {100 * self.recall:.2f}%",
            f" Accuracy: {100 * self.accuracy:.2f}%",
            f" Confusion matrix counts: TP={self.TP}, FP={self.FP}, FN={self.FN}, TN={self.TN}",
        ]
        return "\n".join(lines)


class AccuracyMetrics(BaseModel):
    """Metrics for accuracy evaluation: total records, unambiguous records,
    matches (MM and OO), and accuracy (MM and OO).
    """

    total_records: int = 0
    unambiguous_records: int = 0
    matches_oo: int = 0
    matches_om: int = 0
    matches_mo: int = 0
    matches_mm: int = 0
    accuracy_oo_total: float = 0.0
    accuracy_mm_total: float = 0.0
    accuracy_oo_unambiguous: float = 0.0
    accuracy_om_unambiguous: float = 0.0
    accuracy_mo_unambiguous: float = 0.0

    def report_metrics(self, title: str = "Initial"):
        """Pretty print the accuracy metrics."""
        lines = [
            f"\n{title} classification accuracy metrics:",
            f""" {title} accuracy (OO, subset coded unambiguously by both): {
                100 * self.accuracy_oo_unambiguous:.2f}% ({self.matches_oo} records)""",
            f""" {title} accuracy (OM, subset coded unambiguously by clerical): {
                100 * self.accuracy_om_unambiguous:.2f}% ({self.matches_om} records)""",
            f""" {title} accuracy (MO, subset coded unambiguously by model): {
                100 * self.accuracy_mo_unambiguous:.2f}% ({self.matches_mo} records)""",
            f""" {title} accuracy (MM, full set): {
                100 * self.accuracy_mm_total:.2f}% ({self.matches_mm} records)""",
        ]
        return "\n".join(lines)


class CodabilityMetrics(BaseModel):
    """Metrics for codability: initial and final codable proportions,
    improvement in codability, and counts of codable records.
    """

    initial_codable_prop: float
    final_codable_prop: float | None = None
    codability_improvement_prop: float | None = None
    initial_codable_count: int
    final_codable_count: int | None = None

    def report_metrics(self):
        """Pretty print the codability metrics."""
        lines = [
            "\nCodability metrics:",
            f""" Initial codability: {
                100 * self.initial_codable_prop:.2f}% ({self.initial_codable_count} records)""",
        ]
        if self.final_codable_prop is not None:
            lines.append(
                f""" Final codability: {
                100 * self.final_codable_prop:.2f}% ({self.final_codable_count} records)"""
            )
        if self.codability_improvement_prop is not None:
            lines.append(
                f""" Gain in codability: {100 * self.codability_improvement_prop:.2f}pp ({
                    self.final_codable_count - self.initial_codable_count} records)"""
            )
        return "\n".join(lines)


class SimpleMetrics(BaseModel):
    """Container for all simple evaluation metrics."""

    ambiguity_metrics: AmbiguityMetrics
    codability_metrics: CodabilityMetrics
    initial_accuracy_metrics: AccuracyMetrics
    final_accuracy_metrics: AccuracyMetrics | None = None

    def report_metrics(self):
        """Pretty print all simple metrics."""
        lines = [
            "Evaluation metrics summary:",
            self.ambiguity_metrics.report_metrics(),
            self.codability_metrics.report_metrics(),
            self.initial_accuracy_metrics.report_metrics("Initial"),
        ]
        if self.final_accuracy_metrics:
            lines.append(self.final_accuracy_metrics.report_metrics("Final"))
        return "\n".join(lines)

    def as_dict(self):
        """Return simple metrics as a dictionary."""
        return {
            "ambiguity_metrics": self.ambiguity_metrics.__dict__,
            "codability_metrics": self.codability_metrics.__dict__,
            "initial_accuracy_metrics": self.initial_accuracy_metrics.__dict__,
            "final_accuracy_metrics": (
                self.final_accuracy_metrics.__dict__
                if self.final_accuracy_metrics
                else None
            ),
        }


def calc_ambiguity_metrics(
    df: pd.DataFrame,
    model_ambiguous_col: str = "initial_ambiguous",
    truth_ambiguous_col: str = "clerical_ambiguous",
) -> AmbiguityMetrics:
    """Calculate ambiguity detection metrics: precision, recall, F1-score.

    Args:
        df: DataFrame containing model and clerical ambiguity columns.
        model_ambiguous_col: Column name for model ambiguity predictions (boolean).
        truth_ambiguous_col: Column name for true (clerical) ambiguity labels (boolean).

    Returns:
        Dictionary with precision, recall, and F1-score.
    """
    true_pos = sum(df[model_ambiguous_col] & df[truth_ambiguous_col])
    false_pos = sum(df[model_ambiguous_col] & ~df[truth_ambiguous_col])
    false_neg = sum(~df[model_ambiguous_col] & df[truth_ambiguous_col])
    true_neg = sum(~df[model_ambiguous_col] & ~df[truth_ambiguous_col])

    precision = 0 if true_pos + false_pos == 0 else true_pos / (true_pos + false_pos)
    recall = 0 if true_pos + false_neg == 0 else true_pos / (true_pos + false_neg)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * (precision * recall) / (precision + recall)
    )
    accuracy = (true_pos + true_neg) / len(df) if len(df) > 0 else 0.0

    return AmbiguityMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        TP=true_pos,
        FP=false_pos,
        FN=false_neg,
        TN=true_neg,
    )


def calc_codability_metrics(
    df: pd.DataFrame,
    initial_ambiguous_col: str = "initial_ambiguous",
    final_ambiguous_col: str | None = "final_ambiguous",
) -> CodabilityMetrics:
    """Calculate codability metrics: initial and final codable proportions,
    improvement in codability, and counts of codable records.

    Args:
        df: DataFrame containing model ambiguity columns.
        initial_ambiguous_col: Column name for initial model ambiguity predictions (boolean).
        final_ambiguous_col: Column name for final model ambiguity predictions (boolean).

    Returns:
        Dictionary with codability metrics.
    """
    total_count = len(df)
    initial_codable_count = sum(~df[initial_ambiguous_col])
    initial_codable_prop = (
        initial_codable_count / total_count if total_count > 0 else 0.0
    )

    if final_ambiguous_col and (final_ambiguous_col in df.columns):
        final_codable_count = sum(~df[final_ambiguous_col])
        final_codable_prop = (
            final_codable_count / total_count if total_count > 0 else 0.0
        )
        codability_improvement_prop = (
            (final_codable_count - initial_codable_count) / total_count
            if total_count > 0
            else 0.0
        )
    else:
        final_codable_count = None
        final_codable_prop = None
        codability_improvement_prop = None

    return CodabilityMetrics(
        initial_codable_prop=initial_codable_prop,
        final_codable_prop=final_codable_prop,
        codability_improvement_prop=codability_improvement_prop,
        initial_codable_count=initial_codable_count,
        final_codable_count=final_codable_count,
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
        return AccuracyMetrics()

    def compare_row(row: pd.Series, method) -> bool:
        return _compare_codes(row[truth_col], row[model_col], method=method)

    matches = {}
    for method in ["OO", "OM", "MO", "MM"]:
        matches[method] = sum(df.apply(compare_row, method=method, axis=1))

    unambiguous_om = sum(df[truth_col].apply(len) == 1)
    unambiguous_mo = sum(df[model_col].apply(len) == 1)
    unambiguous_oo = sum(
        (df[truth_col].apply(len) == 1) & (df[model_col].apply(len) == 1)
    )

    accuracy_oo_unambiguous = (
        matches["OO"] / unambiguous_oo if unambiguous_oo > 0 else 0.0
    )
    accuracy_om_unambiguous = (
        matches["OM"] / unambiguous_om if unambiguous_om > 0 else 0.0
    )
    accuracy_mo_unambiguous = (
        matches["MO"] / unambiguous_mo if unambiguous_mo > 0 else 0.0
    )

    return AccuracyMetrics(
        total_records=total,
        unambiguous_records=unambiguous_oo,
        matches_mm=matches["MM"],
        accuracy_mm_total=matches["MM"] / total,
        matches_oo=matches["OO"],
        accuracy_oo_total=matches["OO"] / total,
        accuracy_oo_unambiguous=accuracy_oo_unambiguous,
        matches_om=matches["OM"],
        accuracy_om_unambiguous=accuracy_om_unambiguous,
        matches_mo=matches["MO"],
        accuracy_mo_unambiguous=accuracy_mo_unambiguous,
    )


def calc_simple_metrics(
    df: pd.DataFrame,
    truth_col: str = "clerical_codes",
    initial_model_col: str = "sa_initial_codes",
    final_model_col: str | None = "sa_final_codes",
) -> SimpleMetrics:
    """Calculate ambiguity detection and classification accuracy metrics.

    Args:
        df: DataFrame containing model and clerical code columns.
        truth_col: Column name for true (clerical) codes (string or list/set).
        initial_model_col: Column name for initial model predicted codes.
        final_model_col: Column name for final model predicted codes.
            Defaults to None (no final accuracy metrics calculated).

    Returns:
        Dictionary with calculated metrics.
    """
    if final_model_col and (final_model_col not in df.columns):
        logger.warning(
            "No final classification stats available (final code column not found).",
        )
        final_model_col = None

    required_cols = [initial_model_col, truth_col] + (
        [final_model_col] if final_model_col else []
    )
    if miss := set(required_cols) - set(df.columns):
        raise ValueError(f"DataFrame is missing required columns: {miss}")

    df = df[required_cols].copy()
    for col in df.columns:
        df[col] = df[col].apply(cast_code_to_set)

    df["truth_ambiguous"] = df[truth_col].apply(len) != 1
    df["initial_ambiguous"] = df[initial_model_col].apply(len) != 1

    if final_model_col:
        df["final_ambiguous"] = df[final_model_col].apply(len) != 1

    # Calculate ambiguity metrics
    ambig_metrics = calc_ambiguity_metrics(
        df,
        model_ambiguous_col="initial_ambiguous",
        truth_ambiguous_col="truth_ambiguous",
    )

    # Calculate codability metrics
    codability_metrics = calc_codability_metrics(
        df,
        initial_ambiguous_col="initial_ambiguous",
        final_ambiguous_col="final_ambiguous" if final_model_col else None,
    )

    # Calculate classification accuracy metrics
    initial_accuracy_metrics = calc_accuracy_metrics(
        df, model_col=initial_model_col, truth_col=truth_col
    )

    if final_model_col:
        final_accuracy_metrics = calc_accuracy_metrics(
            df,
            model_col=final_model_col,
            truth_col=truth_col,
        )
    else:
        final_accuracy_metrics = None

    return SimpleMetrics(
        ambiguity_metrics=ambig_metrics,
        codability_metrics=codability_metrics,
        initial_accuracy_metrics=initial_accuracy_metrics,
        final_accuracy_metrics=final_accuracy_metrics,
    )
