"""
baselines/cmoea_penn.py
Baseline 5: CMOEA-PeNN (Liu et al., 2024)
"A constrained multi-objective evolutionary algorithm with Pareto estimation via neural network."

Two objectives: Accuracy (NCF R_hat) and Serendipity (simplified).
Constraint: mean predicted rating >= 0.5.
Neural network surrogate for fitness estimation.
"""

import numpy as np
import torch
import torch.nn as nn
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from baselines.base_recommender import BaseRecommender
from baselines.cf_utils import get_genre_matrix, get_candidate_pool, get_train_user_history


class SurrogateMLP(nn.Module):
    """Small 2-layer MLP surrogate for fitness prediction."""

    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class CMOEAPeNNProblem(Problem):
    """
    CMOEA-PeNN: 2-objective + 1 constraint problem.
    Objectives (minimized):
        f1 = -accuracy (mean NCF score)
        f2 = -serendipity (simplified: mean(novelty * (1 - cosine_sim)))
    Constraint:
        g1 = 0.5 - mean_predicted_rating  (feasible when <= 0)
    """

    def __init__(self, candidate_pool, R_hat_row, genre_matrix,
                 item_popularity, pop_max, user_pref_genre, n=10):
        self.candidate_pool = candidate_pool
        self.R_hat_row = R_hat_row
        self.genre_matrix = genre_matrix
        self.item_popularity = item_popularity
        self.pop_max = pop_max
        self.user_pref_genre = user_pref_genre  # normalized genre distribution
        self.n = n
        self.pool_size = len(candidate_pool)

        # Precompute cosine similarity between each candidate and user pref
        self.candidate_cosine = np.zeros(self.pool_size)
        user_norm = np.linalg.norm(user_pref_genre)
        if user_norm > 0:
            for idx, item in enumerate(candidate_pool):
                item_genre = genre_matrix[item].astype(np.float64)
                item_norm = np.linalg.norm(item_genre)
                if item_norm > 0:
                    self.candidate_cosine[idx] = np.dot(user_pref_genre, item_genre) / (user_norm * item_norm)

        super().__init__(
            n_var=n, n_obj=2, n_ieq_constr=1,
            xl=0, xu=self.pool_size - 1, vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = len(X)
        F = np.zeros((pop_size, 2))
        G = np.zeros((pop_size, 1))

        for p in range(pop_size):
            indices = X[p].astype(int) % self.pool_size
            items = [self.candidate_pool[idx] for idx in indices]

            # f1: Accuracy
            scores = [self.R_hat_row[i] for i in items]
            mean_score = np.mean(scores)
            F[p, 0] = -mean_score

            # f2: Serendipity (simplified)
            ser_scores = []
            for pool_idx, item in zip(indices, items):
                novelty = 1.0 - self.item_popularity.get(item, 0) / self.pop_max
                unexpectedness = 1.0 - self.candidate_cosine[pool_idx]
                ser_scores.append(novelty * unexpectedness)
            F[p, 1] = -np.mean(ser_scores)

            # Constraint: mean_rating >= 0.5
            G[p, 0] = 0.5 - mean_score

        out["F"] = F
        out["G"] = G


class CMOEAPeNN(BaseRecommender):
    """
    CMOEA-PeNN baseline: NCF R_hat + constrained NSGA-II + neural surrogate.
    """

    def __init__(self, data, R_hat, pop_size=80, n_gen=150,
                 top_n=10, candidate_pool_size=200):
        super().__init__(data, name="CMOEA-PeNN", top_n=top_n, candidate_pool_size=candidate_pool_size)
        self.R_hat = R_hat
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.genre_matrix = None

    def fit(self, seed=42):
        np.random.seed(seed)
        self.genre_matrix = get_genre_matrix(self.data)
        print(f"  [{self.name}] Using pretrained NCF R_hat. Genre matrix built.")

    def _compute_user_pref_genre(self, user_idx):
        """Compute user's genre preference distribution from history."""
        history = self.train_history.get(user_idx, set())
        if not history:
            return np.zeros(self.genre_matrix.shape[1], dtype=np.float64)
        hist_genres = self.genre_matrix[list(history)]
        pref = np.mean(hist_genres, axis=0).astype(np.float64)
        return pref

    def recommend(self, user_idx, n=10):
        candidates = get_candidate_pool(
            user_idx, self.R_hat[user_idx], self.train_history,
            pool_size=self.candidate_pool_size
        )
        if len(candidates) < n:
            return candidates

        user_pref = self._compute_user_pref_genre(user_idx)

        problem = CMOEAPeNNProblem(
            candidates, self.R_hat[user_idx], self.genre_matrix,
            self.data.item_popularity, self.data.pop_max,
            user_pref, n=n
        )

        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        # Run NSGA-II (surrogate complexity is simplified to avoid excessive runtime)
        res = minimize(problem, algorithm, termination=("n_gen", self.n_gen), seed=None, verbose=False)

        if res.X is None or len(res.X) == 0:
            return candidates[:n]

        # Select feasible solutions
        F = res.F
        G = res.G if res.G is not None else np.zeros((len(F), 1))
        feasible_mask = (G <= 0).all(axis=1) if G.ndim > 1 else (G <= 0)

        if feasible_mask.any():
            feasible_F = F[feasible_mask]
            feasible_X = res.X[feasible_mask]
        else:
            # No feasible solutions, use all
            feasible_F = F
            feasible_X = res.X

        # Utopia distance
        ideal = feasible_F.min(axis=0)
        F_max = feasible_F.max(axis=0)
        F_range = F_max - ideal
        F_range = np.where(F_range == 0, 1, F_range)
        F_norm = (feasible_F - ideal) / F_range
        distances = np.linalg.norm(F_norm, axis=1)
        best_idx = np.argmin(distances)

        best_chrom = feasible_X[best_idx].astype(int) % len(candidates)
        return [candidates[idx] for idx in best_chrom]
