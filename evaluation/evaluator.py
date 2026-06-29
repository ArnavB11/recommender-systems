"""
evaluation/evaluator.py
Runs evaluation for any recommender at a single cutoff, computing all 10 metrics.
"""

import numpy as np
from evaluation.metrics import (
    precision_at_n, recall_at_n, ndcg_at_n, map_at_n, f1_at_n,
    diversity_score, explainability_score, novelty_score,
    compute_fairness_for_list, compute_serendipity_for_list
)


def evaluate_recommender(recommendations, data, user_indices, R_hat,
                         genre_matrix, test_relevant, train_history,
                         dopm_system, serendipity_system, fairness_model,
                         n=10):
    """
    Evaluate a set of recommendations across all 10 metrics.
    
    Args:
        recommendations: dict[user_idx -> list of item indices]
        data: MovieLensData instance
        user_indices: list of user indices to evaluate
        R_hat: NCF predicted rating matrix (n_users, n_items) — used for fairness
        genre_matrix: np.ndarray (n_items, n_genres)
        test_relevant: dict[user_idx -> set of relevant item indices]
        train_history: dict[user_idx -> set of training item indices]
        dopm_system: fitted DOPMRecommender
        serendipity_system: fitted SerendipityModel
        fairness_model: fitted FairnessObjective
        n: cutoff
    
    Returns:
        dict: metric_name -> float (averaged across users)
    """
    metrics = {
        "Precision": [],
        "Recall": [],
        "NDCG": [],
        "MAP": [],
        "F1": [],
        "Diversity": [],
        "Explainability": [],
        "Novelty": [],
        "Fairness": [],
        "Serendipity": [],
    }

    for u_idx in user_indices:
        rec = recommendations.get(u_idx, [])
        if not rec:
            continue

        relevant = test_relevant.get(u_idx, set())
        history = train_history.get(u_idx, set())

        # Accuracy metrics (only for users with at least 1 relevant item)
        if relevant:
            metrics["Precision"].append(precision_at_n(rec, relevant, n))
            metrics["Recall"].append(recall_at_n(rec, relevant, n))
            metrics["NDCG"].append(ndcg_at_n(rec, relevant, n))
            metrics["MAP"].append(map_at_n(rec, relevant, n))
            metrics["F1"].append(f1_at_n(rec, relevant, n))

        # Beyond-accuracy metrics (for all users)
        metrics["Diversity"].append(diversity_score(rec, genre_matrix, n))
        metrics["Explainability"].append(explainability_score(rec, history, genre_matrix, n))
        metrics["Novelty"].append(novelty_score(rec, data.item_popularity, data.pop_max, n))

        # Fairness (uses existing MSRS models)
        R_hat_row = R_hat[u_idx] if u_idx < len(R_hat) else np.zeros(data.n_items)
        fair = compute_fairness_for_list(
            rec, u_idx, R_hat_row, dopm_system, serendipity_system,
            fairness_model, list(history), n
        )
        metrics["Fairness"].append(fair)

        # Serendipity (uses existing MSRS SerendipityModel)
        ser = compute_serendipity_for_list(rec, history, serendipity_system, n)
        metrics["Serendipity"].append(ser)

    # Average across users
    result = {}
    for key, values in metrics.items():
        result[key] = float(np.mean(values)) if values else 0.0

    return result
