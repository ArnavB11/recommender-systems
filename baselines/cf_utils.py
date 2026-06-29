"""
baselines/cf_utils.py
Shared collaborative filtering utilities, genre vector helpers, and data extraction
functions used by multiple baseline implementations.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os
import sys

# Add project root to path so we can import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_ratings, GENRE_COLS


def build_rating_matrix(data):
    """
    Build a dense user-item rating matrix from training data.
    Extracts only positive interactions (label == 1) from data.train_df,
    and joins with original ratings to get actual rating values.
    
    Returns:
        rating_matrix: np.ndarray of shape (n_users, n_items), 0 for unobserved
        train_interactions: set of (user_idx, item_idx) tuples in training set
    """
    # Get the original ratings for mapping back
    original_ratings = load_ratings()
    rating_lookup = {}
    for _, row in original_ratings.iterrows():
        u_idx = data.user2idx.get(row["user_id"])
        i_idx = data.item2idx.get(row["item_id"])
        if u_idx is not None and i_idx is not None:
            rating_lookup[(u_idx, i_idx)] = row["rating"]

    # Extract positive training interactions
    train_pos = data.train_df[data.train_df["label"] == 1]
    
    rating_matrix = np.zeros((data.n_users, data.n_items), dtype=np.float32)
    train_interactions = set()
    
    for _, row in train_pos.iterrows():
        u = int(row["user_idx"])
        i = int(row["item_idx"])
        rating = rating_lookup.get((u, i), 3.0)  # Default 3 if not found
        rating_matrix[u, i] = rating
        train_interactions.add((u, i))
    
    return rating_matrix, train_interactions


def build_sparse_rating_matrix(data):
    """
    Build a sparse CSR user-item rating matrix from training data.
    More memory-efficient for CF computations.
    """
    original_ratings = load_ratings()
    rating_lookup = {}
    for _, row in original_ratings.iterrows():
        u_idx = data.user2idx.get(row["user_id"])
        i_idx = data.item2idx.get(row["item_id"])
        if u_idx is not None and i_idx is not None:
            rating_lookup[(u_idx, i_idx)] = row["rating"]

    train_pos = data.train_df[data.train_df["label"] == 1]
    
    mat = lil_matrix((data.n_users, data.n_items), dtype=np.float32)
    
    for _, row in train_pos.iterrows():
        u = int(row["user_idx"])
        i = int(row["item_idx"])
        rating = rating_lookup.get((u, i), 3.0)
        mat[u, i] = rating
    
    return mat.tocsr()


def user_cf_predict(rating_matrix, k_neighbors=50):
    """
    User-based CF with adjusted cosine similarity.
    R_hat[u,i] = R_mean[u] + Σ(sim(u,v) * (R[v,i] - R_mean[v])) / Σ|sim(u,v)|
    
    Args:
        rating_matrix: np.ndarray (n_users, n_items), 0 for unobserved
        k_neighbors: number of nearest neighbors
    
    Returns:
        R_hat: np.ndarray (n_users, n_items) predicted ratings
    """
    n_users, n_items = rating_matrix.shape
    
    # Compute user means (over rated items only)
    mask = (rating_matrix > 0).astype(np.float32)
    rating_count = mask.sum(axis=1)
    rating_count = np.where(rating_count == 0, 1, rating_count)
    user_means = (rating_matrix.sum(axis=1)) / rating_count
    
    # Mean-center the ratings (only for observed entries)
    centered = rating_matrix - np.outer(user_means, np.ones(n_items))
    centered = centered * mask  # Zero out unobserved
    
    # Compute user-user cosine similarity on centered ratings
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = centered / norms
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, 0)  # No self-similarity
    
    # Predict ratings
    R_hat = np.full((n_users, n_items), 0.0, dtype=np.float32)
    
    for u in range(n_users):
        sims = sim_matrix[u]
        # Get top-k neighbors
        if k_neighbors < n_users:
            top_k_idx = np.argpartition(sims, -k_neighbors)[-k_neighbors:]
        else:
            top_k_idx = np.arange(n_users)
        
        neighbor_sims = sims[top_k_idx]
        pos_mask = neighbor_sims > 0
        
        if pos_mask.sum() == 0:
            R_hat[u] = user_means[u]
            continue
        
        active_idx = top_k_idx[pos_mask]
        active_sims = neighbor_sims[pos_mask]
        
        # Weighted sum of centered ratings
        numerator = active_sims @ centered[active_idx]
        denominator = np.abs(active_sims).sum()
        
        if denominator > 0:
            R_hat[u] = user_means[u] + numerator / denominator
        else:
            R_hat[u] = user_means[u]
    
    return R_hat


def item_cf_predict(rating_matrix, k_neighbors=50):
    """
    Item-based CF using item-item cosine similarity on the user-item matrix.
    
    Args:
        rating_matrix: np.ndarray (n_users, n_items), 0 for unobserved
        k_neighbors: number of nearest item neighbors
    
    Returns:
        R_hat: np.ndarray (n_users, n_items) predicted ratings
    """
    n_users, n_items = rating_matrix.shape
    mask = (rating_matrix > 0).astype(np.float32)
    
    # Compute item-item cosine similarity (transpose so items are rows)
    item_vecs = rating_matrix.T  # (n_items, n_users)
    norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = item_vecs / norms
    item_sim = normed @ normed.T  # (n_items, n_items)
    np.fill_diagonal(item_sim, 0)
    
    R_hat = np.zeros((n_users, n_items), dtype=np.float32)
    
    for i in range(n_items):
        sims = item_sim[i]
        if k_neighbors < n_items:
            top_k = np.argpartition(sims, -min(k_neighbors, n_items - 1))[-k_neighbors:]
        else:
            top_k = np.arange(n_items)
        
        neighbor_sims = sims[top_k]
        pos_mask = neighbor_sims > 0
        
        if pos_mask.sum() == 0:
            continue
        
        active_idx = top_k[pos_mask]
        active_sims = neighbor_sims[pos_mask]
        
        # For each user, weighted average of their ratings for similar items
        user_ratings_for_neighbors = rating_matrix[:, active_idx]  # (n_users, n_active)
        user_mask_for_neighbors = mask[:, active_idx]
        
        weighted_sum = user_ratings_for_neighbors @ active_sims
        sim_sum = user_mask_for_neighbors @ active_sims
        
        valid = sim_sum > 0
        R_hat[valid, i] = weighted_sum[valid] / sim_sum[valid]
    
    return R_hat


def svd_predict(rating_matrix, rank=50):
    """
    SVD-based rating prediction using TruncatedSVD.
    
    Args:
        rating_matrix: np.ndarray (n_users, n_items)
        rank: number of latent factors
    
    Returns:
        R_hat: np.ndarray (n_users, n_items)
    """
    svd = TruncatedSVD(n_components=rank, random_state=42)
    U = svd.fit_transform(rating_matrix)
    R_hat = U @ svd.components_
    return R_hat.astype(np.float32)


def get_genre_matrix(data):
    """
    Build binary genre matrix of shape (n_items, 19).
    
    Args:
        data: MovieLensData instance
    
    Returns:
        genre_matrix: np.ndarray (n_items, 19)
    """
    n_items = data.n_items
    genre_matrix = np.zeros((n_items, len(GENRE_COLS)), dtype=np.float32)
    
    for item_idx in range(n_items):
        item_id = data.idx2item[item_idx]
        if item_id in data.items_df.index:
            for g_idx, genre in enumerate(GENRE_COLS):
                if genre in data.items_df.columns:
                    genre_matrix[item_idx, g_idx] = float(data.items_df.loc[item_id, genre])
    
    return genre_matrix


def get_test_relevant_items(data):
    """
    Get ground-truth relevant items per user from the test/validation set.
    Relevant = items with original rating >= 4 in the test split.
    
    Returns:
        dict: user_idx -> set of relevant item_idx
    """
    original_ratings = load_ratings()
    rating_lookup = {}
    for _, row in original_ratings.iterrows():
        u_idx = data.user2idx.get(row["user_id"])
        i_idx = data.item2idx.get(row["item_id"])
        if u_idx is not None and i_idx is not None:
            rating_lookup[(u_idx, i_idx)] = row["rating"]

    val_pos = data.val_df[data.val_df["label"] == 1]
    
    relevant = {}
    for _, row in val_pos.iterrows():
        u = int(row["user_idx"])
        i = int(row["item_idx"])
        actual_rating = rating_lookup.get((u, i), 0)
        if actual_rating >= 4:
            relevant.setdefault(u, set()).add(i)
    
    return relevant


def get_train_user_history(data):
    """
    Get training-set user history (items the user interacted with in training).
    
    Returns:
        dict: user_idx -> set of item_idx
    """
    train_pos = data.train_df[data.train_df["label"] == 1]
    history = {}
    for _, row in train_pos.iterrows():
        u = int(row["user_idx"])
        i = int(row["item_idx"])
        history.setdefault(u, set()).add(i)
    return history


def get_unseen_items(user_idx, train_history, n_items):
    """
    Get all unseen items for a user (items not in training history).
    """
    seen = train_history.get(user_idx, set())
    return [i for i in range(n_items) if i not in seen]


def popularity_fallback_topn(data, user_history_set, n=10):
    """
    Cold-start fallback: return top-N globally popular unseen items.
    """
    sorted_by_pop = sorted(data.item_popularity.items(), key=lambda x: -x[1])
    result = []
    for item_idx, _ in sorted_by_pop:
        if item_idx not in user_history_set:
            result.append(item_idx)
            if len(result) >= n:
                break
    return result


def get_candidate_pool(user_idx, R_hat_row, train_history, pool_size=200):
    """
    Get top candidate items for a user by predicted score, excluding seen items.
    
    Returns:
        list of item indices (sorted by descending predicted score)
    """
    seen = train_history.get(user_idx, set())
    unseen_mask = np.ones(len(R_hat_row), dtype=bool)
    for i in seen:
        if i < len(unseen_mask):
            unseen_mask[i] = False
    
    unseen_indices = np.where(unseen_mask)[0]
    if len(unseen_indices) == 0:
        return []
    
    unseen_scores = R_hat_row[unseen_indices]
    top_count = min(pool_size, len(unseen_indices))
    top_pos = np.argpartition(unseen_scores, -top_count)[-top_count:]
    top_pos = top_pos[np.argsort(-unseen_scores[top_pos])]
    
    return unseen_indices[top_pos].tolist()


def normalize_scores(R_hat):
    """
    Normalize predicted rating matrix to [0, 1] range per user.
    """
    mins = R_hat.min(axis=1, keepdims=True)
    maxs = R_hat.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1, ranges)
    return (R_hat - mins) / ranges
