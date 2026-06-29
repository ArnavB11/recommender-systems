"""
baselines/pmoea.py
Baseline 1: PMOEA (Cui et al., 2017)
"A novel multi-objective evolutionary algorithm for recommendation systems."

Two objectives: Accuracy (mean User-CF predicted rating) and Diversity (intra-list genre dissimilarity).
Uses NSGA-II with standard operators.
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from baselines.base_recommender import BaseRecommender
from baselines.cf_utils import (
    build_rating_matrix, user_cf_predict, get_genre_matrix,
    get_candidate_pool, get_train_user_history
)


class PMOEAProblem(Problem):
    """
    PMOEA: 2-objective problem for a single user.
    Objectives (minimized):
        f1 = -mean_predicted_rating (accuracy)
        f2 = -diversity (1 - mean pairwise cosine sim of genre vectors)
    """

    def __init__(self, candidate_pool, R_hat_row, genre_matrix, n=10):
        self.candidate_pool = candidate_pool
        self.R_hat_row = R_hat_row
        self.genre_matrix = genre_matrix
        self.n = n
        self.pool_size = len(candidate_pool)

        # Precompute genre cosine sim for candidate pool items
        pool_genres = genre_matrix[candidate_pool].astype(np.float64)
        norms = np.linalg.norm(pool_genres, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = pool_genres / norms
        self.genre_cos_sim = normed @ normed.T

        super().__init__(
            n_var=n,
            n_obj=2,
            xl=0,
            xu=self.pool_size - 1,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = len(X)
        f1 = np.zeros(pop_size)
        f2 = np.zeros(pop_size)

        for p in range(pop_size):
            indices = X[p].astype(int) % self.pool_size
            items = [self.candidate_pool[idx] for idx in indices]

            # Accuracy: mean predicted rating
            scores = [self.R_hat_row[i] for i in items]
            f1[p] = -np.mean(scores)

            # Diversity: 1 - mean pairwise cosine similarity of genre vectors
            n_items = len(indices)
            if n_items < 2:
                f2[p] = 0.0
            else:
                total_sim = 0.0
                n_pairs = 0
                for i in range(n_items):
                    for j in range(i + 1, n_items):
                        total_sim += self.genre_cos_sim[indices[i], indices[j]]
                        n_pairs += 1
                mean_sim = total_sim / n_pairs if n_pairs > 0 else 0.0
                f2[p] = -(1.0 - mean_sim)  # Maximize diversity

        out["F"] = np.column_stack([f1, f2])


class PMOEA(BaseRecommender):
    """PMOEA baseline: User-CF + NSGA-II (Accuracy + Diversity)."""

    def __init__(self, data, pop_size=100, n_gen=200, top_n=10, candidate_pool_size=200):
        super().__init__(data, name="PMOEA", top_n=top_n, candidate_pool_size=candidate_pool_size)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.R_hat = None
        self.genre_matrix = None

    def fit(self, seed=42):
        np.random.seed(seed)
        print(f"  [{self.name}] Building User-CF predicted ratings...")
        rating_matrix, _ = build_rating_matrix(self.data)
        self.R_hat = user_cf_predict(rating_matrix, k_neighbors=50)
        self.genre_matrix = get_genre_matrix(self.data)
        print(f"  [{self.name}] Fit complete. R_hat shape: {self.R_hat.shape}")

    def recommend(self, user_idx, n=10):
        candidates = get_candidate_pool(
            user_idx, self.R_hat[user_idx], self.train_history,
            pool_size=self.candidate_pool_size
        )
        if len(candidates) < n:
            return candidates

        problem = PMOEAProblem(candidates, self.R_hat[user_idx], self.genre_matrix, n=n)

        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        res = minimize(problem, algorithm, termination=("n_gen", self.n_gen), seed=None, verbose=False)

        if res.X is None or len(res.X) == 0:
            return candidates[:n]

        # Selection: 0.6 * Accuracy + 0.4 * Diversity (on normalized Pareto front)
        F = res.F
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range = np.where(F_range == 0, 1, F_range)
        F_norm = (F - F_min) / F_range

        # Objectives are negated, so lower = better => weight lower
        weighted = 0.6 * F_norm[:, 0] + 0.4 * F_norm[:, 1]
        best_idx = np.argmin(weighted)

        best_chrom = res.X[best_idx].astype(int) % len(candidates)
        return [candidates[idx] for idx in best_chrom]
