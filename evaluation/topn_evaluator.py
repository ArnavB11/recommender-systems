"""
evaluation/topn_evaluator.py
Runs evaluation at multiple top-N cutoffs for generating top-N curve plots.
"""

import numpy as np
from evaluation.evaluator import evaluate_recommender


def evaluate_at_multiple_cutoffs(recommendations_extended, data, user_indices, R_hat,
                                  genre_matrix, test_relevant, train_history,
                                  dopm_system, serendipity_system, fairness_model,
                                  cutoffs=(5, 10, 15, 20, 25, 30)):
    """
    Evaluate recommendations at multiple top-N cutoffs.
    
    Args:
        recommendations_extended: dict[user_idx -> list of item indices (length >= max(cutoffs))]
        data, user_indices, R_hat, genre_matrix, test_relevant, train_history,
        dopm_system, serendipity_system, fairness_model: same as evaluate_recommender
        cutoffs: tuple of N values to evaluate at
    
    Returns:
        dict: metric_name -> dict[cutoff -> float]
        Example: {"Precision": {5: 0.12, 10: 0.09, ...}, ...}
    """
    all_results = {}

    for n in cutoffs:
        # Truncate recommendations to cutoff n
        truncated = {}
        for u_idx, rec_list in recommendations_extended.items():
            truncated[u_idx] = list(rec_list)[:n]

        result = evaluate_recommender(
            truncated, data, user_indices, R_hat,
            genre_matrix, test_relevant, train_history,
            dopm_system, serendipity_system, fairness_model,
            n=n
        )

        for metric_name, value in result.items():
            if metric_name not in all_results:
                all_results[metric_name] = {}
            all_results[metric_name][n] = value

    return all_results
