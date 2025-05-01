"""Module for evaluating classifAI results.

Classes:
    EvaluationResult: Structured evaluation results
    SICAssignmentEvaluator: Evaluate results from API against the validated dataset
    MethodComparison: Compare G-Code and classifAI results
    LabelAccuracy: Analyse classification accuracy for scenarios where model predictions can match any of multiple ground truth labels

"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class EvaluationResult:
    """Structured evaluation results.

    Attributes
    ----------
        mean_rank: Average position of correct answers in results
        total_samples: Total number of test cases evaluated
        found_ratio: Proportion of correct answers found in results
        rank_distribution: Distribution of ranks for found answers
        not_found_count: Number of correct answers not found
        mean_distance: Average distance across all results
        mean_distance_correct: Average distance for correct matches
    """

    mean_rank: float
    total_samples: int
    found_ratio: float
    rank_distribution: Dict[int, int]
    not_found_count: int
    mean_distance: Optional[float] = None
    mean_distance_correct: Optional[float] = None
