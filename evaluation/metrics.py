"""
evaluation/metrics.py
All 10 evaluation metric functions for recommender system comparison.
- Accuracy metrics: Precision@N, Recall@N, NDCG@N, MAP@N, F1@N
- Beyond-accuracy metrics: Diversity, Explainability, Novelty, Fairness, Serendipity
"""

import numpy as np


# =====================================================================
# Accuracy / Ranking Metrics
# =====================================================================

def precision_at_n(recommended, relevant, n=10):
    """
    Precision@N = |relevant ∩ recommended_top_n| / N
    """
    rec = list(recommended)[:n]
    if not rec:
        return 0.0
    hits = len(set(rec) & set(relevant))
    return hits / n


def recall_at_n(recommended, relevant, n=10):
    """
    Recall@N = |relevant ∩ recommended_top_n| / |relevant|
    """
    rec = list(recommended)[:n]
    if not relevant:
        return 0.0
    hits = len(set(rec) & set(relevant))
    return hits / len(relevant)


def ndcg_at_n(recommended, relevant, n=10):
    """
    NDCG@N = DCG@N / IDCG@N
    DCG@N = Σ_{k=1}^{N} rel_k / log2(k+1)   where rel_k ∈ {0,1}
    """
    rec = list(recommended)[:n]
    relevant_set = set(relevant)

    dcg = 0.0
    for k, item in enumerate(rec, start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(k + 1)

    # Ideal DCG: all relevant items at the top
    n_relevant = min(len(relevant_set), n)
    idcg = sum(1.0 / np.log2(k + 1) for k in range(1, n_relevant + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def map_at_n(recommended, relevant, n=10):
    """
    MAP@N = (1/|relevant|) * Σ_{k=1}^{N} P@k * rel_k
    """
    rec = list(recommended)[:n]
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    cumulative_hits = 0
    sum_precision = 0.0

    for k, item in enumerate(rec, start=1):
        if item in relevant_set:
            cumulative_hits += 1
            sum_precision += cumulative_hits / k

    return sum_precision / len(relevant_set)


def f1_at_n(recommended, relevant, n=10):
    """
    F1@N = 2 * Precision@N * Recall@N / (Precision@N + Recall@N)
    """
    p = precision_at_n(recommended, relevant, n)
    r = recall_at_n(recommended, relevant, n)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


# =====================================================================
# Beyond-Accuracy Metrics
# =====================================================================

def diversity_score(recommended, genre_matrix, n=10):
    """
    Mean pairwise genre Jaccard distance between all item pairs in the list.
    Diversity = (2 / (N*(N-1))) * Σ_{i<j} d_jac(i,j)
    
    Args:
        recommended: list of item indices
        genre_matrix: np.ndarray of shape (n_items, n_genres), binary
        n: cutoff
    """
    rec = list(recommended)[:n]
    N = len(rec)
    if N < 2:
        return 0.0

    total_dist = 0.0
    n_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            g_i = set(np.where(genre_matrix[rec[i]] > 0)[0])
            g_j = set(np.where(genre_matrix[rec[j]] > 0)[0])
            if not g_i and not g_j:
                dist = 1.0
            else:
                intersection = len(g_i & g_j)
                union = len(g_i | g_j)
                dist = 1.0 - (intersection / union) if union > 0 else 1.0
            total_dist += dist
            n_pairs += 1

    return total_dist / n_pairs if n_pairs > 0 else 0.0


def explainability_score(recommended, user_history_items, genre_matrix, n=10):
    """
    Mean cosine similarity between each recommended item's genre vector and 
    the user's history genre distribution vector. Normalised to [0,1].
    
    Args:
        recommended: list of item indices
        user_history_items: set/list of item indices in user's training history
        genre_matrix: np.ndarray of shape (n_items, n_genres), binary
        n: cutoff
    """
    rec = list(recommended)[:n]
    if not rec or not user_history_items:
        return 0.0

    # Build user genre distribution: sum of genre vectors of all history items, then normalize
    history_list = list(user_history_items)
    user_genre_dist = np.sum(genre_matrix[history_list], axis=0).astype(np.float64)
    norm_user = np.linalg.norm(user_genre_dist)
    if norm_user == 0:
        return 0.0

    scores = []
    for item_idx in rec:
        item_genre = genre_matrix[item_idx].astype(np.float64)
        norm_item = np.linalg.norm(item_genre)
        if norm_item == 0:
            scores.append(0.0)
        else:
            cos_sim = np.dot(user_genre_dist, item_genre) / (norm_user * norm_item)
            # Clamp to [0, 1] (genre vectors are non-negative so cos_sim >= 0)
            scores.append(float(np.clip(cos_sim, 0.0, 1.0)))

    return float(np.mean(scores))


def novelty_score(recommended, item_popularity, pop_max, n=10):
    """
    Mean (1 - pop(i)/pop_max) for items in list.
    
    Args:
        recommended: list of item indices
        item_popularity: dict mapping item_idx -> rating count
        pop_max: maximum popularity value
        n: cutoff
    """
    rec = list(recommended)[:n]
    if not rec or pop_max == 0:
        return 0.0

    scores = []
    for item_idx in rec:
        pop = item_popularity.get(item_idx, 0)
        scores.append(1.0 - pop / pop_max)

    return float(np.mean(scores))


def compute_fairness_for_list(recommended, user_idx, R_hat_row,
                               dopm_system, serendipity_system, fairness_model,
                               user_history, n=10):
    """
    Compute fairness for a recommendation list using the existing MSRS fairness pipeline.
    
    For baselines, we compute DOPM and serendipity scores for their recommended items
    using the existing fitted models, then pass to FairnessObjective.compute_fairness().
    
    Args:
        recommended: list of item indices
        user_idx: user index
        R_hat_row: NCF predicted rating row for this user (1D array)
        dopm_system: fitted DOPMRecommender
        serendipity_system: fitted SerendipityModel
        fairness_model: fitted FairnessObjective
        user_history: list of item indices in user's history
        n: cutoff
    """
    rec = list(recommended)[:n]
    if not rec:
        return 0.0

    user_pref_vec = serendipity_system.compute_user_preference_vector(set(user_history))

    dopm_dict = {}
    ser_dict = {}
    for m in rec:
        predicted_rating = float(R_hat_row[m]) if m < len(R_hat_row) else 0.0
        dopm_dict[m] = dopm_system.calculate_dopm(user_idx, m, predicted_rating)
        ser_dict[m] = serendipity_system.calculate_serendipity(user_history, user_pref_vec, m)

    return fairness_model.compute_fairness(rec, dopm_dict, ser_dict)


def compute_serendipity_for_list(recommended, user_history, serendipity_system, n=10):
    """
    Compute mean serendipity for a recommendation list using the existing SerendipityModel.
    
    Args:
        recommended: list of item indices
        user_history: list/set of item indices in user's history
        serendipity_system: fitted SerendipityModel
        n: cutoff
    """
    rec = list(recommended)[:n]
    if not rec:
        return 0.0

    history_list = list(user_history)
    user_pref_vec = serendipity_system.compute_user_preference_vector(set(user_history))

    scores = []
    for m in rec:
        s = serendipity_system.calculate_serendipity(history_list, user_pref_vec, m)
        scores.append(s)

    return float(np.mean(scores))
