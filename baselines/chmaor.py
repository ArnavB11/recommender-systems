"""
baselines/chmaor.py
Baseline 2: CHMAOR (Wang & Chen, 2021)
"A novel cascade hybrid many-objective recommendation algorithm 
incorporating multistakeholder concerns."

Three objectives: Accuracy, Novelty, Provider Coverage.
Uses Item-CF + NSGA-III with Das-Dennis reference directions.
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from baselines.base_recommender import BaseRecommender
from baselines.cf_utils import (
    build_rating_matrix, item_cf_predict, get_genre_matrix,
    get_candidate_pool, get_train_user_history
)
from data_loader import GENRE_COLS


class CHMAORProblem(Problem):
    """
    CHMAOR: 3-objective problem.
    Objectives (minimized):
        f1 = -mean_predicted_rating (accuracy)
        f2 = -mean_novelty (1 - pop/pop_max)
        f3 = -provider_coverage (unique genres / total genres)
    """

    def __init__(self, candidate_pool, R_hat_row, genre_matrix,
                 item_popularity, pop_max, total_genres, n=10):
        self.candidate_pool = candidate_pool
        self.R_hat_row = R_hat_row
        self.genre_matrix = genre_matrix
        self.item_popularity = item_popularity
        self.pop_max = pop_max
        self.total_genres = total_genres
        self.n = n
        self.pool_size = len(candidate_pool)

        super().__init__(
            n_var=n, n_obj=3,
            xl=0, xu=self.pool_size - 1, vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = len(X)
        F = np.zeros((pop_size, 3))

        for p in range(pop_size):
            indices = X[p].astype(int) % self.pool_size
            items = [self.candidate_pool[idx] for idx in indices]

            # f1: Accuracy
            scores = [self.R_hat_row[i] for i in items]
            F[p, 0] = -np.mean(scores)

            # f2: Novelty (1 - pop/pop_max)
            novelties = [1.0 - self.item_popularity.get(i, 0) / self.pop_max for i in items]
            F[p, 1] = -np.mean(novelties)

            # f3: Provider Coverage (unique genres / total genres)
            all_genres = set()
            for i in items:
                genres_present = np.where(self.genre_matrix[i] > 0)[0]
                all_genres.update(genres_present)
            coverage = len(all_genres) / self.total_genres if self.total_genres > 0 else 0
            F[p, 2] = -coverage

        out["F"] = F


class CHMAOR(BaseRecommender):
    """CHMAOR baseline: Item-CF + NSGA-III (Accuracy + Novelty + Coverage)."""

    def __init__(self, data, n_gen=200, top_n=10, candidate_pool_size=200):
        super().__init__(data, name="CHMAOR", top_n=top_n, candidate_pool_size=candidate_pool_size)
        self.n_gen = n_gen
        self.R_hat = None
        self.genre_matrix = None
        self.total_genres = len(GENRE_COLS)

    def fit(self, seed=42):
        np.random.seed(seed)
        print(f"  [{self.name}] Building Item-CF predicted ratings...")
        rating_matrix, _ = build_rating_matrix(self.data)
        self.R_hat = item_cf_predict(rating_matrix, k_neighbors=50)
        self.genre_matrix = get_genre_matrix(self.data)
        print(f"  [{self.name}] Fit complete. R_hat shape: {self.R_hat.shape}")

    def recommend(self, user_idx, n=10):
        candidates = get_candidate_pool(
            user_idx, self.R_hat[user_idx], self.train_history,
            pool_size=self.candidate_pool_size
        )
        if len(candidates) < n:
            return candidates

        problem = CHMAORProblem(
            candidates, self.R_hat[user_idx], self.genre_matrix,
            self.data.item_popularity, self.data.pop_max,
            self.total_genres, n=n
        )

        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        pop_size = len(ref_dirs)  # 91

        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        res = minimize(problem, algorithm, termination=("n_gen", self.n_gen), seed=None, verbose=False)

        if res.X is None or len(res.X) == 0:
            return candidates[:n]

        # Selection: highest sum of normalised objectives
        F = res.F
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range = np.where(F_range == 0, 1, F_range)
        F_norm = (F - F_min) / F_range

        # Since objectives are negated (lower=better), sum of normalized = lower is better
        scores = F_norm.sum(axis=1)
        best_idx = np.argmin(scores)

        best_chrom = res.X[best_idx].astype(int) % len(candidates)
        return [candidates[idx] for idx in best_chrom]
